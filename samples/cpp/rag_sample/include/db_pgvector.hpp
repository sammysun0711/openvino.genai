#ifndef _DBPGVECTOR
#define _DBPGVECTOR

#include "state.hpp"

// pgvector-cpp
#include <iostream>
#include <pqxx/pqxx>
#include "pqxx.hpp"
#include <cassert>
#include <optional>

class DBPgvector{
    public:

        pqxx::connection pgconn{"user=postgres host=localhost password=openvino port=5432 dbname=postgres"};

        DBPgvector() = default;
        ~DBPgvector() = default;

        void db_setup();
        void db_store_embeddings(std::vector<std::string> chunks, std::vector<std::vector<float>> embeddings);
        std::vector<std::string> db_retrieval(size_t chunk_size,
                          std::vector<std::string> query,
                          std::vector<std::vector<float>> query_embedding);
};

#endif


