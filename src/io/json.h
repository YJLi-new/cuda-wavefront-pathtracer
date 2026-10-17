// io/json.h - tiny key lookup for the flat JSON files used here (camera / lighting); not a general JSON parser
#pragma once
#include <string>
#include <vector>

std::string readTextFile(const std::string& path);   // "" if unreadable
bool jsonNumber(const std::string& s, const std::string& key, double& out);
// all numbers inside the (possibly nested) array that follows "key"
std::vector<double> jsonNumbers(const std::string& s, const std::string& key);
// 16 numbers of a row-major 4x4 matrix; the key may hold one matrix or a list of per-frame matrices
bool jsonMatrix(const std::string& s, const std::string& key, int frame, float* m16);

// ---- small recursive-descent JSON parser (objects, arrays, numbers, strings, true/false/null) ----
struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false; double num = 0; std::string str;
    std::vector<JsonValue> arr; std::vector<std::pair<std::string, JsonValue>> obj;
    const JsonValue* get(const std::string& key) const;              // object member or nullptr
    double number(const std::string& key, double def) const;         // numeric member or def
    std::vector<double> numbers(const std::string& key) const;       // numeric array member (empty if absent)
    std::string string(const std::string& key, const std::string& def) const;
};
bool parseJson(const std::string& text, JsonValue& out, std::string* err = nullptr);
