#include "full_reference.hpp"

#include <openssl/evp.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
class Hash {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
    Hash() {
        if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("cannot initialize Full cache SHA-256");
    }
    void add(const std::string& bytes) {
        if (EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) != 1)
            throw std::runtime_error("Full cache SHA-256 update failed");
    }
    std::string finish() {
        std::array<unsigned char, 32> bytes{};
        unsigned int size = 0;
        if (EVP_DigestFinal_ex(ctx.get(), bytes.data(), &size) != 1 || size != bytes.size())
            throw std::runtime_error("Full cache SHA-256 finish failed");
        std::ostringstream out;
        for (auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
        return out.str();
    }
};
void put(std::string& bytes, std::uint64_t value, int count = 8) {
    for (int i = 0; i < count; ++i) { bytes.push_back(static_cast<char>(value & 255)); value >>= 8; }
}
std::uint64_t get(const std::string& bytes, std::size_t& at, int count = 8) {
    std::uint64_t value = 0;
    for (int i = 0; i < count; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(at++))) << (i * 8);
    return value;
}
constexpr int limbs = (std::numeric_limits<long double>::digits + 63) / 64;
constexpr std::size_t cellBytes = 20 + 2 * (4 + 8 * limbs);

// Canonical exact binary significand/exponent, no ABI padding and no decimal rounding.
// Covers binary64, x87 extended and binary128. Zero has a canonical exponent of zero.
void putReal(std::string& bytes, long double value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite Full reference");
    int exponent = 0;
    long double fraction = std::frexp(std::abs(value), &exponent);
    if (exponent <= -32768 || exponent >= 32768)
        throw std::runtime_error("unsupported Full reference exponent");
    put(bytes, static_cast<unsigned>(exponent + 32768) | (std::signbit(value) ? 65536U : 0U), 4);
    for (int i = 0; i < limbs; ++i) {
        fraction = std::ldexp(fraction, 64);
        const auto digit = static_cast<std::uint64_t>(fraction);
        put(bytes, digit);
        fraction -= static_cast<long double>(digit);
    }
    if (fraction != 0) throw std::runtime_error("inexact Full reference encoding");
}
long double getReal(const std::string& bytes, std::size_t& at) {
    const auto tag = get(bytes, at, 4);
    if (tag > 131071) throw std::runtime_error("invalid Full reference exponent");
    const int exponent = static_cast<int>(tag & 65535) - 32768;
    long double value = 0;
    for (int i = 0; i < limbs; ++i)
        value += std::ldexp(static_cast<long double>(get(bytes, at)), exponent - 64 * (i + 1));
    return tag & 65536 ? -value : value;
}
void checkRow(const FullReferenceRow& row, int cols) {
    int previous = -1;
    for (const auto& cell : row) {
        if (cell.column <= previous || cell.column >= cols || cell.terms <= 0 ||
            !std::isfinite(cell.serial) || !std::isfinite(cell.reference) ||
            !std::isfinite(cell.bound) || cell.bound < 0)
            throw std::runtime_error("invalid Full reference coordinate");
        previous = cell.column;
    }
}
std::string header(const std::string& key, int rows, int cols) {
    if (rows < 0 || cols < 0 || std::numeric_limits<long double>::radix != 2 || limbs > 2)
        throw std::runtime_error("unsupported Full reference metadata");
    std::string bytes = "SPFR0001" + key;
    put(bytes, rows); put(bytes, cols);
    put(bytes, std::numeric_limits<long double>::digits);
    put(bytes, std::numeric_limits<long double>::max_exponent);
    put(bytes, cellBytes);
    return bytes;
}
// Never unlink the lock inode. OS process death releases this lock, including timeout/kill.
class Lock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    OVERLAPPED overlap{};
#else
    int fd = -1;
#endif
public:
    explicit Lock(const std::filesystem::path& path) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr, OPEN_ALWAYS, 0, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open Full cache lock");
        if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlap)) {
            CloseHandle(handle);
            throw std::runtime_error("Full reference preparation busy or filesystem locking unavailable");
        }
#else
        fd = open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("cannot open Full cache lock");
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            throw std::runtime_error("Full reference preparation busy or filesystem locking unavailable");
        }
#endif
    }
    ~Lock() {
#ifdef _WIN32
        UnlockFileEx(handle, 0, 1, 0, &overlap); CloseHandle(handle);
#else
        close(fd);
#endif
    }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;
};
}

std::string fullReferenceKey(const CsrMatrix& a, const CsrMatrix& b) {
    Hash hash;
    hash.add("full-componentwise_dot_bound_v1-SPFR0001/" + partialReferenceKey(a, b));
    hash.add(header(std::string(64, '0'), a.rows, b.cols));
    return hash.finish();
}
std::filesystem::path fullReferencePath(const std::filesystem::path& directory, const std::string& key) {
    if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("invalid Full reference key");
    return directory / (key + ".sfull");
}

struct FullReferenceReader::Impl {
    std::ifstream file;
    Hash hash;
    int rows, cols, next = 0;
    bool finished = false;
    std::string read(std::size_t size, bool hashed = true) {
        std::string bytes(size, '\0');
        if (!file.read(bytes.data(), static_cast<std::streamsize>(size)))
            throw std::runtime_error("missing/truncated Full cache; prepare references before recovery");
        if (hashed) hash.add(bytes);
        return bytes;
    }
    Impl(const std::filesystem::path& path, const std::string& key, int r, int c)
        : file(path, std::ios::binary), rows(r), cols(c) {
        const auto expected = header(key, rows, cols);
        if (read(expected.size()) != expected)
            throw std::runtime_error("incompatible Full reference header/build/input");
    }
};
FullReferenceReader::FullReferenceReader(const std::filesystem::path& directory, const std::string& key,
                                       int rows, int cols)
    : impl(std::make_unique<Impl>(fullReferencePath(directory, key), key, rows, cols)) {}
FullReferenceReader::~FullReferenceReader() = default;
void FullReferenceReader::readRow(FullReferenceRow& row) {
    if (impl->next >= impl->rows || impl->finished) throw std::runtime_error("Full reference row overflow");
    std::size_t at = 0;
    const auto count = get(impl->read(8), at);
    if (count > static_cast<unsigned>(impl->cols) || count > std::numeric_limits<std::size_t>::max() / cellBytes)
        throw std::runtime_error("invalid Full reference row length");
    const auto bytes = impl->read(static_cast<std::size_t>(count) * cellBytes);
    at = 0; row.clear(); row.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        FullReferenceCell cell;
        const auto column = get(bytes, at, 4), terms = get(bytes, at), serial = get(bytes, at);
        if (column >= static_cast<unsigned>(impl->cols) || terms > std::uint64_t(std::numeric_limits<CsrOffset>::max()))
            throw std::runtime_error("Full reference coordinate out of range");
        cell.column = static_cast<int>(column); cell.terms = static_cast<CsrOffset>(terms);
        std::memcpy(&cell.serial, &serial, sizeof(double));
        cell.reference = getReal(bytes, at); cell.bound = getReal(bytes, at);
        row.push_back(cell);
    }
    checkRow(row, impl->cols);
    ++impl->next;
}
void FullReferenceReader::finish() {
    if (impl->finished || impl->next != impl->rows) throw std::runtime_error("Full reference not completely read");
    if (impl->read(64, false) != impl->hash.finish() || impl->file.peek() != std::char_traits<char>::eof() || impl->file.bad())
        throw std::runtime_error("corrupt Full reference checksum/trailing data");
    impl->finished = true;
}

void prepareFullReferenceFile(const std::filesystem::path& directory, const std::string& key,
                             int rows, int cols, const std::function<void(int, FullReferenceRow&)>& buildRow,
                             ValidationDiagnostics& d) {
    std::filesystem::create_directories(directory);
    const auto path = fullReferencePath(directory, key);
    Lock lock(path.string() + ".lock");
    FullReferenceRow row;
    if (std::filesystem::exists(path)) {
        const auto start = Clock::now();
        FullReferenceReader reader(directory, key, rows, cols);
        for (int i = 0; i < rows; ++i) reader.readRow(row);
        reader.finish();
        d.readSeconds = seconds(start); d.source = "reused";
        return;
    }
    const auto temporary = path.string() + ".tmp";
    const auto closer = [](FILE* f) { std::fclose(f); };
    std::unique_ptr<FILE, decltype(closer)> file(std::fopen(temporary.c_str(), "wb"), closer);
    if (!file) throw std::runtime_error("cannot create Full reference temporary file");
    Hash hash;
    auto write = [&](const std::string& bytes, bool hashed = true) {
        if (std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size())
            throw std::runtime_error("cannot write Full reference (check disk space/quota)");
        if (hashed) hash.add(bytes);
    };
    write(header(key, rows, cols));
    for (int i = 0; i < rows; ++i) {
        auto start = Clock::now();
        buildRow(i, row);
        d.buildSeconds += seconds(start);
        start = Clock::now();
        checkRow(row, cols);
        std::string bytes;
        bytes.reserve(8 + row.size() * cellBytes);
        put(bytes, row.size());
        for (const auto& cell : row) {
            put(bytes, static_cast<unsigned>(cell.column), 4); put(bytes, cell.terms);
            std::uint64_t serial;
            std::memcpy(&serial, &cell.serial, sizeof(double));
            put(bytes, serial); putReal(bytes, cell.reference); putReal(bytes, cell.bound);
        }
        write(bytes);
        d.writeSeconds += seconds(start);
    }
    write(hash.finish(), false);
    if (std::fflush(file.get()) != 0) throw std::runtime_error("cannot flush Full reference");
#ifdef _WIN32
    if (_commit(_fileno(file.get())) != 0) throw std::runtime_error("cannot sync Full reference");
#else
    if (fsync(fileno(file.get())) != 0) throw std::runtime_error("cannot sync Full reference");
#endif
    if (std::fclose(file.release()) != 0) throw std::runtime_error("cannot close Full reference");
    std::filesystem::rename(temporary, path);
#ifndef _WIN32
    const int fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("cannot open Full reference directory for sync");
    const int status = fsync(fd); close(fd);
    if (status != 0) throw std::runtime_error("cannot sync Full reference directory");
#endif
    d.source = "generated";
}
