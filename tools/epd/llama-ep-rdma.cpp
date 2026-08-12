// llama-ep-rdma: RDMA (RoCEv2) backend for the EP transport function table.
//
// Connection setup goes through rdma_cm (automatic GID/path resolution, works
// for RoCEv2 on Ethernet link-layer devices). Data path is RDMA Send/Receive
// over a reliable-connected QP with pre-registered ring buffers on both sides.
//
// The framing layer (llama_ep_send_frame/recv_frame) depends on blocking
// byte-stream semantics, so this backend re-assembles a byte stream on top of
// message-oriented Send/Receive: every RDMA Send carries [u32 payload_len]
// followed by up to SLOT_PAYLOAD bytes; the receiver keeps a cursor across
// message boundaries. Chunk boundaries are invisible to the caller.
//
// Completion waiting uses a completion channel (ibv_get_cq_event) instead of
// spinning, so idle wait costs about the same CPU as a blocking TCP recv().

#include "llama-ep-transport.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <cstdlib>

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sched.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

namespace {

const size_t SLOT_PAYLOAD = 256 * 1024;        // max payload bytes per RDMA Send
const size_t SLOT_BYTES   = 4 + SLOT_PAYLOAD;  // u32 len + payload
const int    N_SEND = 4;                       // outstanding sends before ring reuse
// A 16 MiB receive ring keeps a full GLM-5.2 top-8 ubatch=64 RESP2 in flight.
// This matters when another endpoint finishes while the master is draining the
// first one; a smaller ring RNR-stalls the worker and rejects practical PP.
const int    N_RECV = 64;                      // 64 x 256 KiB = 16 MiB payload

void set_err(std::string * err, const char * what) {
    if (err) {
        *err = std::string(what) + ": " + strerror(errno);
    }
}

void set_err_msg(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
}

struct rdma_conn {
    // Each established connection owns a CM event channel.  Keeping accepted
    // connections on the listener channel lets the accept loop consume their
    // DISCONNECTED events, leaving the connection thread blocked on its CQ.
    rdma_event_channel * ec      = nullptr;
    bool                 owns_ec = false;
    rdma_cm_id         * id      = nullptr;
    ibv_pd             * pd      = nullptr;
    // split completion paths: the send side (rdma_send_all) and the recv
    // side (rdma_recv_all) may run on different threads (async pipelined EP
    // keeps a background receiver draining RESP frames while the compute
    // thread fires REQs), so each owns its own CQ + channel + ring state
    ibv_comp_channel   * ch_s    = nullptr;
    ibv_cq             * cq_s    = nullptr;
    ibv_comp_channel   * ch_r    = nullptr;
    ibv_cq             * cq_r    = nullptr;

    uint8_t * sbuf = nullptr;  // N_SEND slots
    ibv_mr  * smr  = nullptr;
    uint8_t * rbuf = nullptr;  // N_RECV slots
    ibv_mr  * rmr  = nullptr;

    bool sslot_free[N_SEND];
    bool rslot_ready[N_RECV];
    int  sends_posted = 0;

    // receive byte-stream cursor
    int      rhead      = 0;      // next slot index to consume (mod N_RECV)
    uint32_t cur_left   = 0;      // payload bytes left in the current message
    size_t   cur_off    = 0;      // absolute offset of cursor into rbuf
    bool     cur_active = false;

    std::atomic<bool> broken {false};
};

bool post_recv(rdma_conn * c, int slot) {
    uint8_t * b = c->rbuf + (size_t) slot * SLOT_BYTES;
    ibv_sge sge = {(uintptr_t) b, (uint32_t) SLOT_BYTES, c->rmr->lkey};
    ibv_recv_wr wr = {};
    wr.wr_id   = (uint64_t) slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    ibv_recv_wr * bad = nullptr;
    if (ibv_post_recv(c->id->qp, &wr, &bad) != 0) {
        c->broken = true;
        return false;
    }
    return true;
}

// returns false on CQE error
bool handle_wcs(rdma_conn * c, ibv_wc * wcs, int n) {
    for (int i = 0; i < n; ++i) {
        if (wcs[i].status != IBV_WC_SUCCESS) {
            // IBV_WC_WR_FLUSH_ERR is how a peer disconnect surfaces on pending recvs
            c->broken = true;
            return false;
        }
        if (wcs[i].wr_id >= 64) {
            c->sslot_free[wcs[i].wr_id - 64] = true;
        } else {
            c->rslot_ready[wcs[i].wr_id] = true;
        }
    }
    return true;
}

// GGML_EP_RDMA_SPIN=1: debug knob — busy-poll the CQ instead of the comp channel
bool spin_mode() {
    static const bool v = []() {
        const char * e = getenv("GGML_EP_RDMA_SPIN");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// Drain one CQ.  A blocking wait watches both the completion channel and the
// connection's CM channel: a peer which exits without a protocol shutdown is
// not guaranteed to generate a CQE for an already-posted receive, but it does
// generate RDMA_CM_EVENT_DISCONNECTED.
bool pump(rdma_conn * c, ibv_cq * cq, ibv_comp_channel * ch, bool block) {
    ibv_wc wcs[16];
    for (;;) {
        int n = ibv_poll_cq(cq, 16, wcs);
        if (n < 0) {
            c->broken = true;
            return false;
        }
        if (n > 0) {
            if (!handle_wcs(c, wcs, n)) {
                return false;
            }
            return true;
        }
        if (c->broken || !block) {
            return !c->broken;
        }
        if (spin_mode()) {
            sched_yield();
            continue;
        }
        if (ibv_req_notify_cq(cq, 0) != 0) {
            c->broken = true;
            return false;
        }
        // re-poll after arming to close the missed-event race
        n = ibv_poll_cq(cq, 16, wcs);
        if (n < 0) {
            c->broken = true;
            return false;
        }
        if (n > 0) {
            if (!handle_wcs(c, wcs, n)) {
                return false;
            }
            return true;
        }
        pollfd pfds[2] = {
            {ch->fd,    POLLIN, 0},
            {c->ec->fd, POLLIN, 0},
        };
        const int pr = poll(pfds, 2, -1);
        if (pr < 0 && errno != EINTR) {
            c->broken = true;
            return false;
        }
        if (pr < 0) {
            continue;
        }

        // CM events after ESTABLISHED are terminal for this byte stream.  The
        // channel is non-blocking because the send and receive pumps can wake
        // on the same event concurrently; only one of them should consume it.
        if (pfds[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) {
            rdma_cm_event * ev = nullptr;
            if (rdma_get_cm_event(c->ec, &ev) == 0) {
                rdma_ack_cm_event(ev);
                c->broken = true;
                return false;
            }
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                c->broken = true;
                return false;
            }
            if (c->broken) {
                return false;
            }
        }
        if (pfds[0].revents & POLLIN) {
            ibv_cq * ev_cq = nullptr;
            void   * ev_ctx = nullptr;
            if (ibv_get_cq_event(ch, &ev_cq, &ev_ctx) == 0) {
                ibv_ack_cq_events(cq, 1);
            }
        } else if (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            c->broken = true;
            return false;
        }
        // loop back and poll
    }
}

bool rdma_sendv_all(void * vctx, const void * const * parts, const size_t * part_lens, size_t n_parts) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    size_t part = 0;
    size_t part_off = 0;
    while (part < n_parts) {
        while (part < n_parts && part_off == part_lens[part]) {
            part++;
            part_off = 0;
        }
        if (part == n_parts) {
            break;
        }
        const int slot = c->sends_posted % N_SEND;
        while (!c->sslot_free[slot]) {
            if (!pump(c, c->cq_s, c->ch_s, true)) {
                return false;
            }
        }
        uint8_t * b = c->sbuf + (size_t) slot * SLOT_BYTES;
        size_t used = 0;
        while (part < n_parts && used < SLOT_PAYLOAD) {
            const size_t left = part_lens[part] - part_off;
            const size_t n = left < SLOT_PAYLOAD - used ? left : SLOT_PAYLOAD - used;
            if (n > 0) {
                memcpy(b + 4 + used, static_cast<const uint8_t *>(parts[part]) + part_off, n);
                used += n;
                part_off += n;
            }
            if (part_off == part_lens[part]) {
                part++;
                part_off = 0;
            }
        }

        const uint32_t hdr = (uint32_t) used;
        memcpy(b, &hdr, 4);

        ibv_sge sge = {(uintptr_t) b, (uint32_t) (4 + used), c->smr->lkey};
        ibv_send_wr wr = {};
        wr.wr_id      = (uint64_t) (64 + slot);
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.opcode     = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ibv_send_wr * bad = nullptr;
        if (ibv_post_send(c->id->qp, &wr, &bad) != 0) {
            c->broken = true;
            return false;
        }
        c->sslot_free[slot] = false;
        c->sends_posted++;
    }
    return true;
}

bool rdma_send_all(void * vctx, const void * data, size_t len) {
    const void * parts[1] = {data};
    const size_t part_lens[1] = {len};
    return rdma_sendv_all(vctx, parts, part_lens, len > 0 ? 1 : 0);
}

bool rdma_recv_all(void * vctx, void * data, size_t len) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    uint8_t * out = static_cast<uint8_t *>(data);
    while (len > 0) {
        if (!c->cur_active) {
            const int slot = c->rhead % N_RECV;
            while (!c->rslot_ready[slot]) {
                if (!pump(c, c->cq_r, c->ch_r, true)) {
                    return false;
                }
            }
            c->rslot_ready[slot] = false;
            uint32_t mlen = 0;
            memcpy(&mlen, c->rbuf + (size_t) slot * SLOT_BYTES, 4);
            if (mlen > SLOT_PAYLOAD) {
                c->broken = true; // corrupt stream
                return false;
            }
            c->cur_left   = mlen;
            c->cur_off    = (size_t) slot * SLOT_BYTES + 4;
            c->cur_active = true;
        }
        const size_t n = len < c->cur_left ? len : c->cur_left;
        memcpy(out, c->rbuf + c->cur_off, n);
        c->cur_off  += n;
        c->cur_left -= (uint32_t) n;
        out += n;
        len -= n;
        if (c->cur_left == 0) {
            const int slot = c->rhead % N_RECV;
            c->cur_active = false;
            c->rhead++;
            // repost only now: the slot is fully consumed
            if (!post_recv(c, slot)) {
                return false;
            }
        }
    }
    return true;
}

void rdma_conn_shutdown(void * vctx) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    c->broken = true;
    if (c->id) {
        // The per-connection CM channel wakes a recv pump blocked in poll().
        rdma_disconnect(c->id);
    }
}

void rdma_conn_close(void * vctx) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    if (c->id) {
        rdma_disconnect(c->id);
    }
    if (c->smr) {
        ibv_dereg_mr(c->smr);
    }
    if (c->rmr) {
        ibv_dereg_mr(c->rmr);
    }
    free(c->sbuf);
    free(c->rbuf);
    if (c->id) {
        rdma_destroy_ep(c->id); // destroys the QP too (created via rdma_create_qp)
    }
    if (c->cq_s) {
        ibv_destroy_cq(c->cq_s);
    }
    if (c->ch_s) {
        ibv_destroy_comp_channel(c->ch_s);
    }
    if (c->cq_r) {
        ibv_destroy_cq(c->cq_r);
    }
    if (c->ch_r) {
        ibv_destroy_comp_channel(c->ch_r);
    }
    if (c->pd) {
        ibv_dealloc_pd(c->pd);
    }
    if (c->owns_ec && c->ec) {
        rdma_destroy_event_channel(c->ec);
    }
    delete c;
}

// wait for one cm event of the expected type
int64_t now_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// wait for one cm event of the expected type; timeout_ms < 0 waits forever.
// A stale peer previously made the master hang inside rdma_get_cm_event during
// warmup with no recovery — bounded waits let the caller fall back to TCP.
bool cm_wait(rdma_event_channel * ec, rdma_cm_event_type want, rdma_cm_id ** id_out, std::string * err,
             int timeout_ms = 5000) {
    const int64_t deadline = now_ms() + (timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
        if (timeout_ms >= 0) {
            const int64_t remain = deadline - now_ms();
            if (remain <= 0) {
                set_err_msg(err, std::string("rdma_cm event wait timed out (") +
                        std::to_string(timeout_ms) + " ms), wanted " + rdma_event_str(want));
                return false;
            }
            pollfd pfd = {ec->fd, POLLIN, 0};
            const int pr = poll(&pfd, 1, (int) remain);
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;
                }
                set_err(err, "poll(cm channel)");
                return false;
            }
            if (pr == 0) {
                continue; // re-check deadline
            }
        }
        rdma_cm_event * ev = nullptr;
        if (rdma_get_cm_event(ec, &ev) != 0) {
            if (errno == EINTR || errno == EAGAIN) {
                continue;
            }
            set_err(err, "rdma_get_cm_event");
            return false;
        }
        const bool ok = ev->event == want;
        if (!ok) {
            set_err_msg(err, "rdma_cm event " + std::to_string((int) ev->event) +
                    " (" + rdma_event_str(ev->event) + "), wanted " + rdma_event_str(want));
        }
        if (id_out) {
            *id_out = ev->id;
        }
        rdma_ack_cm_event(ev);
        return ok;
    }
}

bool make_cm_channel_nonblocking(rdma_event_channel * ec, std::string * err) {
    const int flags = fcntl(ec->fd, F_GETFL, 0);
    if (flags < 0 || fcntl(ec->fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        set_err(err, "fcntl(rdma_cm channel, O_NONBLOCK)");
        return false;
    }
    return true;
}

// Tighten the RNR retry timer on an established RC QP.  The rdma_cm default
// min_rnr_timer is huge (~80 ms scale): a single RNR NAK — inevitable once a
// bulk frame overruns the 8-slot receive ring — stalled the stream by ~80 ms
// per chunk (measured ~77 ms/chunk on 6 MB GLM prefill frames, i.e. ~3 MB/s,
// with multi-second collapse cascades).  0.01 ms recovery keeps an occasional
// RNR at line rate (measured 16 MB frames: 5.5 GB/s, zero >100 ms stalls).
// RTS->RTS accepts only a driver-specific attribute subset (mlx5: just
// min_rnr_timer), so failure degrades to a warning.
void tune_qp(ibv_qp * qp) {
    ibv_qp_attr attr = {};
    attr.qp_state      = IBV_QPS_RTS;
    attr.min_rnr_timer = 1; // 0.01 ms RNR retry delay
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_MIN_RNR_TIMER) != 0) {
        fprintf(stderr, "llama-ep-rdma: tune min_rnr_timer failed: %s (ignored)\n",
                strerror(errno));
    }
}

// create PD/CQ/QP and register the rings on an id that has addr/route resolved
bool setup_conn(rdma_conn * c, std::string * err) {
    c->pd = ibv_alloc_pd(c->id->verbs);
    if (!c->pd) {
        set_err(err, "ibv_alloc_pd");
        return false;
    }
    c->ch_s = ibv_create_comp_channel(c->id->verbs);
    c->ch_r = ibv_create_comp_channel(c->id->verbs);
    if (!c->ch_s || !c->ch_r) {
        set_err(err, "ibv_create_comp_channel");
        return false;
    }
    c->cq_s = ibv_create_cq(c->id->verbs, N_SEND + 16, nullptr, c->ch_s, 0);
    c->cq_r = ibv_create_cq(c->id->verbs, N_RECV + 16, nullptr, c->ch_r, 0);
    if (!c->cq_s || !c->cq_r) {
        set_err(err, "ibv_create_cq");
        return false;
    }

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq_s;
    qia.recv_cq = c->cq_r;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr  = N_SEND + 8;
    qia.cap.max_recv_wr  = N_RECV + 8;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    if (rdma_create_qp(c->id, c->pd, &qia) != 0) {
        set_err(err, "rdma_create_qp");
        return false;
    }

    c->sbuf = static_cast<uint8_t *>(aligned_alloc(4096, N_SEND * SLOT_BYTES));
    c->rbuf = static_cast<uint8_t *>(aligned_alloc(4096, N_RECV * SLOT_BYTES));
    if (!c->sbuf || !c->rbuf) {
        set_err_msg(err, "ring buffer allocation failed");
        return false;
    }
    c->smr = ibv_reg_mr(c->pd, c->sbuf, N_SEND * SLOT_BYTES, IBV_ACCESS_LOCAL_WRITE);
    c->rmr = ibv_reg_mr(c->pd, c->rbuf, N_RECV * SLOT_BYTES, IBV_ACCESS_LOCAL_WRITE);
    if (!c->smr || !c->rmr) {
        set_err(err, "ibv_reg_mr");
        return false;
    }

    for (int i = 0; i < N_SEND; ++i) {
        c->sslot_free[i] = true;
    }
    for (int i = 0; i < N_RECV; ++i) {
        c->rslot_ready[i] = false;
        if (!post_recv(c, i)) {
            set_err(err, "ibv_post_recv");
            return false;
        }
    }
    return true;
}

} // namespace

llama_ep_transport * llama_ep_rdma_connect(const char * host, int port, std::string * err) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * res = nullptr;
    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        set_err(err, "getaddrinfo");
        return nullptr;
    }

    auto * c = new rdma_conn;
    c->owns_ec = true;
    c->ec = rdma_create_event_channel();
    if (!c->ec) {
        set_err(err, "rdma_create_event_channel");
        freeaddrinfo(res);
        delete c;
        return nullptr;
    }
    if (!make_cm_channel_nonblocking(c->ec, err)) {
        freeaddrinfo(res);
        rdma_destroy_event_channel(c->ec);
        delete c;
        return nullptr;
    }
    if (rdma_create_id(c->ec, &c->id, nullptr, RDMA_PS_TCP) != 0) {
        set_err(err, "rdma_create_id (no RDMA device?)");
        freeaddrinfo(res);
        rdma_destroy_event_channel(c->ec);
        delete c;
        return nullptr;
    }

    bool ok = false;
    for (addrinfo * ai = res; ai != nullptr && !ok; ai = ai->ai_next) {
        if (rdma_resolve_addr(c->id, nullptr, ai->ai_addr, 2000) != 0) {
            continue;
        }
        ok = true;
    }
    freeaddrinfo(res);

    rdma_conn_param cparam = {};
    cparam.initiator_depth     = 8;
    cparam.responder_resources = 8;
    cparam.retry_count         = 7;
    cparam.rnr_retry_count     = 7;

    if (ok) ok = cm_wait(c->ec, RDMA_CM_EVENT_ADDR_RESOLVED, nullptr, err);
    if (ok && rdma_resolve_route(c->id, 2000) != 0) {
        set_err(err, "rdma_resolve_route");
        ok = false;
    }
    if (ok) ok = cm_wait(c->ec, RDMA_CM_EVENT_ROUTE_RESOLVED, nullptr, err);
    if (ok) ok = setup_conn(c, err);
    if (ok && rdma_connect(c->id, &cparam) != 0) {
        set_err(err, "rdma_connect");
        ok = false;
    }
    if (ok) ok = cm_wait(c->ec, RDMA_CM_EVENT_ESTABLISHED, nullptr, err);
    if (ok) tune_qp(c->id->qp);

    if (!ok) {
        rdma_conn_close(c);
        return nullptr;
    }

    auto * t = new llama_ep_transport;
    t->ctx = c;
    t->ops = {rdma_send_all, rdma_recv_all, rdma_conn_shutdown, rdma_conn_close, rdma_sendv_all};
    return t;
}

namespace {

struct rdma_listener {
    rdma_event_channel * ec  = nullptr;
    rdma_cm_id         * id  = nullptr;
    int                  port = 0;
};

bool rdma_listener_accept(void * vctx, llama_ep_transport * out) {
    rdma_listener * l = static_cast<rdma_listener *>(vctx);

    rdma_cm_id * conn_id = nullptr;
    std::string err;
    if (!cm_wait(l->ec, RDMA_CM_EVENT_CONNECT_REQUEST, &conn_id, &err, -1)) {
        return false;
    }

    auto * c = new rdma_conn;
    c->id = conn_id;
    c->owns_ec = true;
    c->ec = rdma_create_event_channel();
    if (!c->ec || !make_cm_channel_nonblocking(c->ec, &err) || rdma_migrate_id(conn_id, c->ec) != 0) {
        rdma_conn_close(c);
        return false;
    }
    if (!setup_conn(c, &err)) {
        rdma_conn_close(c);
        return false;
    }

    rdma_conn_param cparam = {};
    cparam.initiator_depth     = 8;
    cparam.responder_resources = 8;
    cparam.retry_count         = 7;
    cparam.rnr_retry_count     = 7;
    if (::rdma_accept(conn_id, &cparam) != 0) {
        rdma_conn_close(c);
        return false;
    }
    if (!cm_wait(c->ec, RDMA_CM_EVENT_ESTABLISHED, nullptr, &err, -1)) {
        rdma_conn_close(c);
        return false;
    }
    tune_qp(conn_id->qp);

    out->ctx = c;
    out->ops = {rdma_send_all, rdma_recv_all, rdma_conn_shutdown, rdma_conn_close, rdma_sendv_all};
    return true;
}

void rdma_listener_close(void * vctx) {
    rdma_listener * l = static_cast<rdma_listener *>(vctx);
    if (l->id) {
        rdma_destroy_ep(l->id);
    }
    if (l->ec) {
        rdma_destroy_event_channel(l->ec);
    }
    delete l;
}

} // namespace

llama_ep_listener * llama_ep_rdma_listen(const char * host, int port, std::string * err) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_PASSIVE;

    addrinfo * res = nullptr;
    if (getaddrinfo(host && host[0] ? host : nullptr, port_str, &hints, &res) != 0) {
        set_err(err, "getaddrinfo");
        return nullptr;
    }

    auto * l = new rdma_listener;
    l->ec = rdma_create_event_channel();
    if (!l->ec) {
        set_err(err, "rdma_create_event_channel");
        freeaddrinfo(res);
        delete l;
        return nullptr;
    }
    if (rdma_create_id(l->ec, &l->id, nullptr, RDMA_PS_TCP) != 0) {
        set_err(err, "rdma_create_id (no RDMA device?)");
        freeaddrinfo(res);
        rdma_destroy_event_channel(l->ec);
        delete l;
        return nullptr;
    }

    bool ok = false;
    for (addrinfo * ai = res; ai != nullptr && !ok; ai = ai->ai_next) {
        if (rdma_bind_addr(l->id, ai->ai_addr) == 0) {
            ok = true;
        }
    }
    freeaddrinfo(res);
    if (!ok) {
        set_err(err, "rdma_bind_addr");
        rdma_listener_close(l);
        return nullptr;
    }
    if (rdma_listen(l->id, 8) != 0) {
        set_err(err, "rdma_listen");
        rdma_listener_close(l);
        return nullptr;
    }
    l->port = ntohs(rdma_get_src_port(l->id));

    auto * out = new llama_ep_listener;
    out->ctx = l;
    out->ops = {rdma_listener_accept, rdma_listener_close};
    return out;
}

int llama_ep_rdma_listener_port(const llama_ep_listener * l) {
    return static_cast<rdma_listener *>(l->ctx)->port;
}
