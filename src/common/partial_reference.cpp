#include "partial_reference.hpp"
#include "reference_build.hpp"

#include <openssl/evp.h>
#include <array>
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
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
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

class Sha256 {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(), EVP_MD_CTX_free};
    std::array<unsigned char, 65536> buffer{};
    std::size_t used = 0;
    void flush() {
        if (used && EVP_DigestUpdate(context.get(), buffer.data(), used) != 1)
            throw std::runtime_error("SHA-256 update failed");
        used = 0;
    }
public:
    Sha256() {
        if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("cannot initialize SHA-256");
    }
    void add(const void* data, std::size_t size) {
        flush();
        if (EVP_DigestUpdate(context.get(), data, size) != 1)
            throw std::runtime_error("SHA-256 update failed");
    }
    void add(const std::string& s) { add(s.data(), s.size()); }
    void integer(std::uint64_t value) {
        if (used == buffer.size()) flush();
        for (int i = 0; i < 8; ++i) { buffer[used++] = static_cast<unsigned char>(value); value >>= 8; }
    }
    std::string finish() {
        flush();
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int size = 0;
        if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != 32)
            throw std::runtime_error("SHA-256 finalization failed");
        std::ostringstream out;
        for (unsigned int i = 0; i < size; ++i)
            out << std::hex << std::setfill('0') << std::setw(2) << unsigned(digest[i]);
        return out.str();
    }
};
std::string sha256(const std::string& bytes) { Sha256 hash; hash.add(bytes); return hash.finish(); }

std::string matrixHash(const CsrMatrix& matrix) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "reference identity requires IEEE-754 binary64");
    Sha256 hash;
    hash.add("ordered-csr-le64-binary64-v1");
    hash.integer(matrix.rows); hash.integer(matrix.cols);
    hash.integer(matrix.rowPtr.size());
    for (auto v : matrix.rowPtr) hash.integer(static_cast<std::uint64_t>(v));
    hash.integer(matrix.columnIndices.size());
    for (auto v : matrix.columnIndices) hash.integer(static_cast<std::uint64_t>(v));
    hash.integer(matrix.values.size());
    for (double v : matrix.values) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        hash.integer(bits);
    }
    return hash.finish();
}

void integer(std::string& bytes, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { bytes.push_back(static_cast<char>(value & 255)); value >>= 8; }
}
std::uint64_t integer(const std::string& bytes, std::size_t& at) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(at++))) << (8 * i);
    return value;
}
void checkKey(const std::string& key) {
    if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("invalid partial reference key");
}

// Kernel-owned, nonblocking lock. Never unlink the lock inode: process death releases it.
class GeneratorLock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    OVERLAPPED overlap{};
#else
    int fd = -1;
#endif
public:
    explicit GeneratorLock(const std::filesystem::path& path) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, 0, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open reference lock");
        if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlap)) {
            CloseHandle(handle);
            throw std::runtime_error("reference generator busy (or locking unavailable); retry later");
        }
#else
        fd = open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("cannot open reference lock: " + path.string());
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            throw std::runtime_error("reference generator busy (or locking unavailable); retry later");
        }
#endif
    }
    ~GeneratorLock() {
#ifdef _WIN32
        UnlockFileEx(handle, 0, 1, 0, &overlap); CloseHandle(handle);
#else
        close(fd);
#endif
    }
    GeneratorLock(const GeneratorLock&) = delete;
    GeneratorLock& operator=(const GeneratorLock&) = delete;
};

void publish(const std::filesystem::path& path, const std::string& bytes) {
    // One writer owns this pathname under the lock. An interrupted .tmp is truncated on retry.
    const auto temporary = path.string() + ".tmp";
    const auto closeFile = [](FILE* file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(closeFile)> file(std::fopen(temporary.c_str(), "wb"), closeFile);
    if (!file || std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size() ||
        std::fflush(file.get()) != 0) throw std::runtime_error("cannot write partial reference");
#ifdef _WIN32
    if (_commit(_fileno(file.get())) != 0) throw std::runtime_error("cannot sync partial reference");
#else
    if (fsync(fileno(file.get())) != 0) throw std::runtime_error("cannot sync partial reference");
#endif
    if (std::fclose(file.release()) != 0) throw std::runtime_error("cannot close partial reference");
    std::filesystem::rename(temporary, path);
#ifndef _WIN32
    const int fd = open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("cannot open reference directory for sync");
    const int status = fsync(fd);
    close(fd);
    if (status != 0) throw std::runtime_error("cannot sync reference directory");
#endif
}
}  // namespace

std::string partialReferenceBuildId() {
    return sha256(std::string(SPGEMM_REFERENCE_BUILD_ID) + "/" + SPGEMM_REFERENCE_CONFIGURATION);
}

std::string partialReferenceKey(const CsrMatrix& a, const CsrMatrix& b) {
    Sha256 hash;
    // Reader retains row-local input order, duplicates and explicit zeros, and expands symmetry.
    hash.add("partial-v1/mm-coordinate-order-duplicates-zeros-preserved/numeric-serial-exact-zero-drop/");
    hash.add(partialReferenceBuildId());
    hash.integer(static_cast<std::uint64_t>(std::fegetround()));
#if defined(__SSE__)
    hash.integer(_mm_getcsr() & 0x8040U);  // FTZ/DAZ, excluding transient exception flags.
#else
    hash.integer(0);
#endif
    hash.add(matrixHash(a)); hash.add(matrixHash(b));
    return hash.finish();
}

std::filesystem::path partialReferencePath(const std::filesystem::path& directory, const std::string& key) {
    checkKey(key);
    return directory / (key + ".spref");
}

std::string encodePartialReference(const PartialReference& ref, const std::string& key) {
    checkKey(key);
    if (ref.rows < 0 || ref.cols < 0 || ref.rowNnz.size() != static_cast<std::size_t>(ref.rows))
        throw std::runtime_error("invalid reference dimensions");
    std::string bytes = "SPPR0001" + key;
    integer(bytes, ref.rows); integer(bytes, ref.cols); integer(bytes, ref.finite ? 1 : 0);
    for (auto count : ref.rowNnz) {
        if (count < 0) throw std::runtime_error("negative reference count");
        integer(bytes, static_cast<std::uint64_t>(count));
    }
    return bytes + sha256(bytes);
}

PartialReference decodePartialReference(const std::string& bytes, const std::string& key, int rows, int cols) {
    checkKey(key);
    const auto size = 160ULL + 8ULL * static_cast<unsigned>(rows);
    if (rows < 0 || cols < 0 || bytes.size() != size || bytes.compare(0, 8, "SPPR0001") != 0 ||
        bytes.compare(8, 64, key) != 0 || sha256(bytes.substr(0, bytes.size() - 64)) != bytes.substr(bytes.size() - 64))
        throw std::runtime_error("incomplete, corrupt or incompatible partial reference");
    std::size_t at = 72;
    const auto storedRows = integer(bytes, at), storedCols = integer(bytes, at), finite = integer(bytes, at);
    if (storedRows != static_cast<unsigned>(rows) || storedCols != static_cast<unsigned>(cols) || finite > 1)
        throw std::runtime_error("invalid partial reference metadata");
    PartialReference ref{rows, cols, finite == 1, {}};
    ref.rowNnz.reserve(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const auto count = integer(bytes, at);
        if (count > static_cast<std::uint64_t>(std::numeric_limits<CsrOffset>::max()))
            throw std::runtime_error("partial reference count exceeds signed 64-bit range");
        ref.rowNnz.push_back(static_cast<CsrOffset>(count));
    }
    return ref;
}

PartialReference readPartialReference(const std::filesystem::path& directory, const std::string& key,
                                     int rows, int cols) {
    const auto path = partialReferencePath(directory, key);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto expected = 160ULL + 8ULL * static_cast<unsigned>(rows);
    if (rows < 0 || !input || input.tellg() != static_cast<std::streamoff>(expected))
        throw std::runtime_error("missing/incomplete partial reference; prepare on a compute node: " + path.string());
    std::string bytes(static_cast<std::size_t>(expected), '\0');
    input.seekg(0);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("cannot read partial reference: " + path.string());
    return decodePartialReference(bytes, key, rows, cols);
}

PartialReference preparePartialReference(const CsrMatrix& a, const CsrMatrix& b,
                                        const std::filesystem::path& directory, ValidationDiagnostics& d) {
    auto start = Clock::now();
    d.key = partialReferenceKey(a, b); d.identitySeconds = elapsed(start);
    std::filesystem::create_directories(directory);
    const auto path = partialReferencePath(directory, d.key);
    GeneratorLock lock(path.string() + ".lock");
    if (std::filesystem::exists(path)) {
        start = Clock::now();
        auto ref = readPartialReference(directory, d.key, a.rows, b.cols);
        d.readSeconds = elapsed(start); d.source = "reused";
        return ref;
    }
    start = Clock::now();
    auto ref = buildPartialReference(a, b);
    d.buildSeconds = elapsed(start); d.source = "generated";
    start = Clock::now();
    publish(path, encodePartialReference(ref, d.key));
    d.writeSeconds = elapsed(start);
    return ref;
}

void printValidationDiagnostics(const ValidationDiagnostics& d) {
    std::ostringstream line;
    line << std::setprecision(17) << "validation_reference_source=" << d.source
         << " validation_reference_key=" << (d.key.empty() ? "NA" : d.key)
         << " validation_reference_build=" << partialReferenceBuildId()
         << " validation_identity_seconds=" << d.identitySeconds
         << " validation_reference_build_seconds=" << d.buildSeconds
         << " validation_cache_read_seconds=" << d.readSeconds
         << " validation_compare_seconds=" << d.compareSeconds
         << " validation_cache_write_seconds=" << d.writeSeconds << '\n';
    std::cout << line.str();
}
