// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _DBPGVECTOR
#define _DBPGVECTOR

#include "state.hpp"

#include <cassert>
#include <iostream>
#include <optional>
#include <pqxx/pqxx>

#include "pqxx.hpp"

class DBPgvector {
public:
    DBPgvector() = default;
    ~DBPgvector() = default;

    void db_setup(const std::string& db_connection_str);
    void db_store_embeddings(std::vector<std::string> chunks, std::vector<std::vector<float>> embeddings);
    std::vector<std::string> db_retrieval(size_t chunk_size,
                                          std::vector<std::string> query,
                                          std::vector<std::vector<float>> query_embedding);

private:
    std::optional<pqxx::connection> pgconn;
};

#endif
