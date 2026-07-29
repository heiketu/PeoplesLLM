#include "llama-ep-transport.h"

#include <cerrno>
#include <cstring>

#ifdef _WIN32
#error "llama-ep-transport: POSIX only for now"
#endif

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

void set_err(std::string * err, const char * what) {
    if (err) {
        *err = std::string(what) + ": " + strerror(errno);
    }
}

struct tcp_conn {
    int fd = -1;
};

bool tcp_send_all(void * vctx, const void * data, size_t len) {
    tcp_conn * c = static_cast<tcp_conn *>(vctx);
    const char * p = static_cast<const char *>(data);
    while (len > 0) {
        ssize_t n = send(c->fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p += n;
        len -= (size_t) n;
    }
    return true;
}

bool tcp_recv_all(void * vctx, void * data, size_t len) {
    tcp_conn * c = static_cast<tcp_conn *>(vctx);
    char * p = static_cast<char *>(data);
    while (len > 0) {
        ssize_t n = recv(c->fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false; // EOF
        }
        p += n;
        len -= (size_t) n;
    }
    return true;
}

void tcp_conn_close(void * vctx) {
    tcp_conn * c = static_cast<tcp_conn *>(vctx);
    if (c->fd >= 0) {
        ::close(c->fd);
    }
    delete c;
}

void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

llama_ep_transport * wrap_conn(int fd) {
    set_nodelay(fd);
    auto * c = new tcp_conn{fd};
    auto * t = new llama_ep_transport;
    t->ctx = c;
    t->ops = {tcp_send_all, tcp_recv_all, tcp_conn_close};
    return t;
}

struct tcp_listener {
    int fd = -1;
    int port = 0;
};

bool tcp_accept(void * vctx, llama_ep_transport * out) {
    tcp_listener * l = static_cast<tcp_listener *>(vctx);
    sockaddr_storage addr;
    socklen_t addrlen = sizeof(addr);
    int fd;
    do {
        fd = ::accept(l->fd, reinterpret_cast<sockaddr *>(&addr), &addrlen);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        return false;
    }
    llama_ep_transport * t = wrap_conn(fd);
    *out = *t;
    delete t;
    return true;
}

void tcp_listener_close(void * vctx) {
    tcp_listener * l = static_cast<tcp_listener *>(vctx);
    if (l->fd >= 0) {
        ::close(l->fd);
    }
    delete l;
}

} // namespace

llama_ep_transport * llama_ep_tcp_connect(const char * host, int port, std::string * err) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * res = nullptr;
    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        set_err(err, "getaddrinfo");
        return nullptr;
    }

    int fd = -1;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        set_err(err, "connect");
        return nullptr;
    }
    return wrap_conn(fd);
}

llama_ep_listener * llama_ep_tcp_listen(const char * host, int port, std::string * err) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo * res = nullptr;
    if (getaddrinfo(host && host[0] ? host : nullptr, port_str, &hints, &res) != 0) {
        set_err(err, "getaddrinfo");
        return nullptr;
    }

    int fd = -1;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 16) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        set_err(err, "bind/listen");
        return nullptr;
    }

    auto * l = new tcp_listener;
    l->fd = fd;

    sockaddr_storage addr;
    socklen_t addrlen = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addrlen) == 0) {
        if (addr.ss_family == AF_INET) {
            l->port = ntohs(reinterpret_cast<sockaddr_in *>(&addr)->sin_port);
        } else if (addr.ss_family == AF_INET6) {
            l->port = ntohs(reinterpret_cast<sockaddr_in6 *>(&addr)->sin6_port);
        }
    }

    auto * out = new llama_ep_listener;
    out->ctx = l;
    out->ops = {tcp_accept, tcp_listener_close};
    return out;
}

int llama_ep_tcp_listener_port(const llama_ep_listener * l) {
    return static_cast<tcp_listener *>(l->ctx)->port;
}

bool llama_ep_send_frame(llama_ep_transport * t, uint32_t type, const void * payload, size_t payload_len) {
    const void * parts[1] = {payload};
    const size_t lens[1] = {payload_len};
    return llama_ep_send_framev(t, type, parts, lens, payload ? 1 : 0);
}

bool llama_ep_send_framev(llama_ep_transport * t, uint32_t type, const void ** parts, const size_t * part_lens, size_t n_parts) {
    llama_ep_frame_header hdr;
    hdr.magic = LLAMA_EP_MAGIC;
    hdr.type = type;
    hdr.payload_len = 0;
    for (size_t i = 0; i < n_parts; ++i) {
        hdr.payload_len += part_lens[i];
    }
    if (!t->ops.send_all(t->ctx, &hdr, sizeof(hdr))) {
        return false;
    }
    for (size_t i = 0; i < n_parts; ++i) {
        if (!t->ops.send_all(t->ctx, parts[i], part_lens[i])) {
            return false;
        }
    }
    return true;
}

bool llama_ep_recv_frame(llama_ep_transport * t, uint32_t & type, std::vector<uint8_t> & payload) {
    llama_ep_frame_header hdr;
    if (!t->ops.recv_all(t->ctx, &hdr, sizeof(hdr))) {
        return false;
    }
    if (hdr.magic != LLAMA_EP_MAGIC) {
        return false;
    }
    // sanity cap: 1 GiB per frame
    if (hdr.payload_len > (uint64_t) 1 << 30) {
        return false;
    }
    payload.resize((size_t) hdr.payload_len);
    if (hdr.payload_len > 0 && !t->ops.recv_all(t->ctx, payload.data(), payload.size())) {
        return false;
    }
    type = hdr.type;
    return true;
}
