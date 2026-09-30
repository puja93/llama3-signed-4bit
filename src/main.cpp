#include "llama_engine.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    std::string model_path = "models/llama3_signed_4bit_b128.bin";
    if (argc > 1 && argv[1][0] != '-') {
        model_path = argv[1];
    }

    std::cout << "\033[1;36m";
    std::cout << "================================================================================\n";
    std::cout << "  LLaMA-3.2 1B INSTRUCT - APPLE SILICON COMPACT SIGNED 4-BIT ENGINE\n";
    std::cout << "  macOS (Apple Silicon M-Series) | ARM NEON High-Throughput SIMD\n";
    std::cout << "  Compact 4.25 Bits/Weight (626 MB Model) | Direct Fast Memory-Map\n";
    std::cout << "================================================================================\033[0m\n";

    std::cout << "[1/2] Loading Tokenizer from models/tokenizer.json...\n";
    llama_tokenizer_t tokenizer = llama_tokenizer_create("models/tokenizer.json");
    if (!tokenizer) {
        std::cerr << "Error: Failed to load models/tokenizer.json\n";
        return 1;
    }

    std::cout << "[2/2] Memory-Mapping Model from " << model_path << "...\n";
    auto t_start_load = std::chrono::high_resolution_clock::now();

    llama_engine_t engine = llama_engine_create(model_path.c_str());
    if (!engine) {
        std::cerr << "Error: Failed to load model from " << model_path << "\n";
        llama_tokenizer_free(tokenizer);
        return 1;
    }

    auto t_end_load = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t_end_load - t_start_load).count();

    auto encode = [&](const std::string& text) -> std::vector<int32_t> {
        std::vector<int32_t> buf(4096);
        int n = llama_tokenizer_encode(tokenizer, text.c_str(), buf.data(), static_cast<int>(buf.size()));
        buf.resize(n);
        return buf;
    };

    auto reset_history = [&]() {
        llama_engine_reset(engine);
        std::vector<int32_t> init_hist;
        init_hist.push_back(128000); // <|begin_of_text|>
        init_hist.push_back(128006); // <|start_header_id|>
        for (int32_t id : encode("system")) init_hist.push_back(id);
        init_hist.push_back(128007); // <|end_header_id|>
        for (int32_t id : encode("\n\n")) init_hist.push_back(id);
        for (int32_t id : encode("You are a helpful, concise, and smart AI assistant.")) init_hist.push_back(id);
        init_hist.push_back(128009); // <|eot_id|>
        return init_hist;
    };

    std::cout << "\n================================================================================\n";
    std::cout << "  Model Loaded & Ready in " << std::fixed << std::setprecision(3) << load_sec << "s! (626.34 MB Model Active)\n";
    std::cout << "  Context Window: 8,192 Tokens Active | Pure In-Register SIMD GEMV.\n";
    std::cout << "  Type your prompt and press Enter.\n";
    std::cout << "  Commands: '/reset' or '/clear' to reset chat, '/exit' or '/quit' to leave.\n";
    std::cout << "================================================================================\n\n";

    std::vector<int32_t> chat_history = reset_history();
    size_t processed_pos = 0;
    size_t max_seq_len = static_cast<size_t>(llama_engine_get_max_seq_len(engine));

    std::string user_input;
    while (true) {
        std::cout << "\033[1;32mYou>\033[0m ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input.empty()) continue;

        if (user_input == "/exit" || user_input == "/quit") break;
        if (user_input == "/reset" || user_input == "/clear") {
            chat_history = reset_history();
            processed_pos = 0;
            std::cout << "\033[1;33mChat history reset.\033[0m\n\n";
            continue;
        }

        chat_history.push_back(128006); // <|start_header_id|>
        for (int32_t id : encode("user")) chat_history.push_back(id);
        chat_history.push_back(128007); // <|end_header_id|>
        for (int32_t id : encode("\n\n")) chat_history.push_back(id);
        for (int32_t id : encode(user_input)) chat_history.push_back(id);
        chat_history.push_back(128009); // <|eot_id|>

        chat_history.push_back(128006); // <|start_header_id|>
        for (int32_t id : encode("assistant")) chat_history.push_back(id);
        chat_history.push_back(128007); // <|end_header_id|>
        for (int32_t id : encode("\n\n")) chat_history.push_back(id);

        if (processed_pos + 1 < chat_history.size()) {
            size_t count = (chat_history.size() - 1) - processed_pos;
            llama_engine_prefill(engine, &chat_history[processed_pos], count, processed_pos, 0);
            processed_pos = chat_history.size() - 1;
        }

        std::cout << "\033[1;35mLlama3-Signed-626MB>\033[0m ";
        std::cout.flush();

        auto t0_gen = std::chrono::high_resolution_clock::now();
        size_t gen_tokens = 0;

        while (processed_pos < max_seq_len) {
            llama_engine_forward_token(engine, chat_history[processed_pos], processed_pos, 1);
            int32_t next_token = llama_engine_sample(engine, 0.0f, 0.9f); // Greedy

            if (next_token == 128001 || next_token == 128009) { // <|end_of_text|> or <|eot_id|>
                processed_pos++;
                break;
            }

            const char* piece = llama_tokenizer_decode(tokenizer, next_token);
            std::cout << piece;
            std::cout.flush();

            chat_history.push_back(next_token);
            processed_pos++;
            gen_tokens++;

            if (gen_tokens >= 512) break;
        }

        auto t1_gen = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t1_gen - t0_gen).count();
        double tok_s = (elapsed_sec > 0.0) ? (static_cast<double>(gen_tokens) / elapsed_sec) : 0.0;

        std::cout << "\n\033[0;37m[" << gen_tokens << " tokens | "
                  << std::fixed << std::setprecision(1) << tok_s
                  << " tok/s | Compact Signed 4-Bit (626MB)]\033[0m\n\n";
    }

    llama_engine_free(engine);
    llama_tokenizer_free(tokenizer);
    return 0;
}
