#include "rlprof/profiler/parser.h"

#include <sqlite3.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/profiler/categorizer.h"

namespace rlprof::profiler {
namespace {

constexpr const char* kKernelQuery = R"SQL(
SELECT
    shortName,
    SUM("end" - start) AS total_ns,
    COUNT(*) AS calls,
    AVG("end" - start) AS avg_ns,
    MIN("end" - start) AS min_ns,
    MAX("end" - start) AS max_ns,
    MAX(registersPerThread) AS registers,
    MAX(staticSharedMemory + dynamicSharedMemory) AS shared_mem
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY total_ns DESC
)SQL";

constexpr const char* kKernelQueryWithStringIds = R"SQL(
SELECT
    COALESCE(names.value, CAST(kernels.shortName AS TEXT)) AS shortName,
    SUM(kernels."end" - kernels.start) AS total_ns,
    COUNT(*) AS calls,
    AVG(kernels."end" - kernels.start) AS avg_ns,
    MIN(kernels."end" - kernels.start) AS min_ns,
    MAX(kernels."end" - kernels.start) AS max_ns,
    MAX(kernels.registersPerThread) AS registers,
    MAX(kernels.staticSharedMemory + kernels.dynamicSharedMemory) AS shared_mem
FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
LEFT JOIN StringIds AS names
    ON names.id = kernels.shortName
GROUP BY COALESCE(names.value, CAST(kernels.shortName AS TEXT))
ORDER BY total_ns DESC
)SQL";

using SqliteDbPtr = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
using SqliteStmtPtr = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

std::int64_t column_int64(sqlite3_stmt* statement, int index) {
  return sqlite3_column_int64(statement, index);
}

std::string column_text(sqlite3_stmt* statement, int index) {
  const unsigned char* text = sqlite3_column_text(statement, index);
  return text == nullptr ? std::string() : reinterpret_cast<const char*>(text);
}

bool table_exists(sqlite3* db, const char* table_name) {
  sqlite3_stmt* raw_statement = nullptr;
  constexpr const char* kSql =
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1";
  if (sqlite3_prepare_v2(db, kSql, -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  SqliteStmtPtr statement(raw_statement, sqlite3_finalize);
  if (sqlite3_bind_text(statement.get(), 1, table_name, -1, SQLITE_STATIC) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  const int step_rc = sqlite3_step(statement.get());
  if (step_rc != SQLITE_ROW && step_rc != SQLITE_DONE) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  return step_rc == SQLITE_ROW;
}

}  // namespace

std::vector<KernelRecord> parse_nsys_sqlite(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("SQLite file does not exist: " + path.string());
  }

  sqlite3* raw_db = nullptr;
  const int open_rc = sqlite3_open_v2(
      path.c_str(),
      &raw_db,
      SQLITE_OPEN_READONLY,
      nullptr);
  if (open_rc != SQLITE_OK) {
    const std::string message =
        raw_db == nullptr ? "failed to open sqlite database"
                          : sqlite3_errmsg(raw_db);
    if (raw_db != nullptr) {
      sqlite3_close(raw_db);
    }
    throw std::runtime_error(message);
  }
  SqliteDbPtr db(raw_db, sqlite3_close);

  sqlite3_stmt* raw_statement = nullptr;
  const char* query = table_exists(db.get(), "StringIds")
                          ? kKernelQueryWithStringIds
                          : kKernelQuery;
  const int prepare_rc = sqlite3_prepare_v2(db.get(), query, -1, &raw_statement, nullptr);
  if (prepare_rc != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db.get()));
  }
  SqliteStmtPtr statement(raw_statement, sqlite3_finalize);

  std::vector<KernelRecord> records;
  while (true) {
    const int step_rc = sqlite3_step(statement.get());
    if (step_rc == SQLITE_DONE) {
      break;
    }
    if (step_rc != SQLITE_ROW) {
      throw std::runtime_error(sqlite3_errmsg(db.get()));
    }

    const std::string name = column_text(statement.get(), 0);
    records.push_back(KernelRecord{
        .name = name,
        .category = std::string(categorize(name)),
        .total_ns = column_int64(statement.get(), 1),
        .calls = column_int64(statement.get(), 2),
        .avg_ns = column_int64(statement.get(), 3),
        .min_ns = column_int64(statement.get(), 4),
        .max_ns = column_int64(statement.get(), 5),
        .registers = column_int64(statement.get(), 6),
        .shared_mem = column_int64(statement.get(), 7),
    });
  }

  return records;
}

}  // namespace rlprof::profiler
