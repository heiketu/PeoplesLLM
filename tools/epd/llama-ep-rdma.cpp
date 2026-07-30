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
const int    N_RECV = 8;                       // pre-posted receive slots

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
    rdma_event_channel * ec      = nullptr; // borrowed (listener conns share the listener's channel)
    bool                 owns_ec = false;
    rdma_cm_id         * id      = nullptr;
    ibv_pd             * pd      = nullptr;
    ibv_comp_channel   * ch      = nullptr;
    ibv_cq             * cq      = nullptr;

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

    bool broken = false;
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

// drain the CQ; if block, sleep on the completion channel until at least one CQE
bool pump(rdma_conn * c, bool block) {
    ibv_wc wcs[16];
    for (;;) {
        int n = ibv_poll_cq(c->cq, 16, wcs);
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
        if (ibv_req_notify_cq(c->cq, 0) != 0) {
            c->broken = true;
            return false;
        }
        // re-poll after arming to close the missed-event race
        n = ibv_poll_cq(c->cq, 16, wcs);
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
        pollfd pfd = {c->ch->fd, POLLIN, 0};
        if (poll(&pfd, 1, -1) < 0 && errno != EINTR) {
            c->broken = true;
            return false;
        }
        if (pfd.revents & POLLIN) {
            ibv_cq * ev_cq = nullptr;
            void   * ev_ctx = nullptr;
            if (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) == 0) {
                ibv_ack_cq_events(c->cq, 1);
            }
        }
        // loop back and poll
    }
}

bool rdma_send_all(void * vctx, const void * data, size_t len) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    const uint8_t * p = static_cast<const uint8_t *>(data);
    while (len > 0) {
        const size_t n = len < SLOT_PAYLOAD ? len : SLOT_PAYLOAD;
        const int slot = c->sends_posted % N_SEND;
        while (!c->sslot_free[slot]) {
            if (!pump(c, true)) {
                return false;
            }
        }
        uint8_t * b = c->sbuf + (size_t) slot * SLOT_BYTES;
        const uint32_t hdr = (uint32_t) n;
        memcpy(b, &hdr, 4);
        memcpy(b + 4, p, n);

        ibv_sge sge = {(uintptr_t) b, (uint32_t) (4 + n), c->smr->lkey};
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
        p   += n;
        len -= n;
    }
    return true;
}

bool rdma_recv_all(void * vctx, void * data, size_t len) {
    rdma_conn * c = static_cast<rdma_conn *>(vctx);
    uint8_t * out = static_cast<uint8_t *>(data);
    while (len > 0) {
        if (!c->cur_active) {
            const int slot = c->rhead % N_RECV;
            while (!c->rslot_ready[slot]) {
                if (!pump(c, true)) {
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
    if (c->cq) {
        ibv_destroy_cq(c->cq);
    }
    if (c->ch) {
        ibv_destroy_comp_channel(c->ch);
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
bool cm_wait(rdma_event_channel * ec, rdma_cm_event_type want, rdma_cm_id ** id_out, std::string * err) {
    rdma_cm_event * ev = nullptr;
    if (rdma_get_cm_event(ec, &ev) != 0) {
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

// Tighten retry timers on an established RC QP.  Defaults chosen by rdma_cm are
// huge (min_rnr_timer can be ~80 ms): a single RNR NAK — inevitable once a bulk
// frame overruns the 8-slot receive ring — then stalls the stream by ~80 ms per
// chunk (measured ~77 ms/chunk on 6 MB GLM prefill frames, i.e. ~3 MB/s).
// RTS->RTS only accepts a subset of attributes depending on the driver, so each
// knob is tried independently and failure degrades to a warning.
void tune_one(ibv_qp * qp, int which, uint32_t value, const char * name) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    switch (which) {
    case IBV_QP_TIMEOUT:        attr.timeout       = value; break;
    case IBV_QP_RETRY_CNT:      attr.retry_cnt     = value; break;
    case IBV_QP_RNR_RETRY:      attr.rnr_retry     = value; break;
    case IBV_QP_MIN_RNR_TIMER:  attr.min_rnr_timer = value; break;
    }
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | which) != 0) {
        fprintf(stderr, "llama-ep-rdma: tune_qp %s=%u failed: %s (ignored)\n",
                name, value, strerror(errno));
    }
}

bool tune_qp(ibv_qp * qp, std::string * err) {
    (void) err;
    tune_one(qp, IBV_QP_MIN_RNR_TIMER, 1, "min_rnr_timer"); // 0.01 ms
    tune_one(qp, IBV_QP_TIMEOUT,       14, "timeout");      // ~67 ms
    tune_one(qp, IBV_QP_RETRY_CNT,     7, "retry_cnt");     // infinite
    tune_one(qp, IBV_QP_RNR_RETRY,     7, "rnr_retry");     // infinite
    return true;
}

// create PD/CQ/QP and register the rings on an id that has addr/route resolved
bool setup_conn(rdma_conn * c, std::string * err) {
    c->pd = ibv_alloc_pd(c->id->verbs);
    if (!c->pd) {
        set_err(err, "ibv_alloc_pd");
        return false;
    }
    c->ch = ibv_create_comp_channel(c->id->verbs);
    if (!c->ch) {
        set_err(err, "ibv_create_comp_channel");
        return false;
    }
    c->cq = ibv_create_cq(c->id->verbs, N_SEND + N_RECV + 16, nullptr, c->ch, 0);
    if (!c->cq) {
        set_err(err, "ibv_create_cq");
        return false;
    }

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
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
    if (ok) ok = tune_qp(c->id->qp, err);

    if (!ok) {
        rdma_conn_close(c);
        return nullptr;
    }

    auto * t = new llama_ep_transport;
    t->ctx = c;
    t->ops = {rdma_send_all, rdma_recv_all, rdma_conn_close};
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
    if (!cm_wait(l->ec, RDMA_CM_EVENT_CONNECT_REQUEST, &conn_id, &err)) {
        return false;
    }

    auto * c = new rdma_conn;
    c->ec = l->ec; // borrowed; listener owns the channel
    c->id = conn_id;
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
    if (!cm_wait(l->ec, RDMA_CM_EVENT_ESTABLISHED, nullptr, &err)) {
        rdma_conn_close(c);
        return false;
    }
    if (!tune_qp(conn_id->qp, &err)) {
        rdma_conn_close(c);
        return false;
    }

    out->ctx = c;
    out->ops = {rdma_send_all, rdma_recv_all, rdma_conn_close};
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
