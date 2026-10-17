// io/json.cpp (written from scratch)
#include "io/json.h"
#include <cstdio>
#include <cstdlib>

std::string readTextFile(const std::string& p) { FILE* fp = fopen(p.c_str(), "rb"); if (!fp) return ""; std::string s; char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) s.append(buf, n); fclose(fp); return s; }

bool jsonNumber(const std::string& s, const std::string& key, double& out) {
    size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) return false; k = s.find(':', k); if (k == std::string::npos) return false;
    out = atof(s.c_str() + k + 1); return true;
}

std::vector<double> jsonNumbers(const std::string& s, const std::string& key) {
    std::vector<double> nums; size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) return nums; k = s.find('[', k); if (k == std::string::npos) return nums;
    int depth = 0;
    for (size_t i = k; i < s.size(); i++) {
        char c = s[i];
        if (c == '[') depth++; else if (c == ']') { depth--; if (depth == 0) break; }
        else if ((c >= '0' && c <= '9') || c == '-' || c == '.') { char* e; nums.push_back(strtod(s.c_str() + i, &e)); i = (e - s.c_str()) - 1; }
    }
    return nums;
}

bool jsonMatrix(const std::string& s, const std::string& key, int frame, float* m16) {
    std::vector<double> nums = jsonNumbers(s, key);
    if (nums.size() < 16) return false;
    size_t off = (size_t)frame * 16; if (nums.size() == 16) off = 0; if (off + 16 > nums.size()) { fprintf(stderr, "frame %d out of range\n", frame); return false; }
    for (int j = 0; j < 16; j++) m16[j] = (float)nums[off + j];
    return true;
}

// ---- JsonValue ----
const JsonValue* JsonValue::get(const std::string& key) const {
    if (type != Object) return nullptr;
    for (auto& kv : obj) if (kv.first == key) return &kv.second;
    return nullptr;
}
double JsonValue::number(const std::string& key, double def) const { const JsonValue* v = get(key); return v && v->type == Number ? v->num : def; }
std::vector<double> JsonValue::numbers(const std::string& key) const {
    std::vector<double> r; const JsonValue* v = get(key); if (!v || v->type != Array) return r;
    for (auto& e : v->arr) if (e.type == Number) r.push_back(e.num);
    return r;
}
std::string JsonValue::string(const std::string& key, const std::string& def) const { const JsonValue* v = get(key); return v && v->type == String ? v->str : def; }

namespace {
struct JsonParser {
    const std::string& s; size_t i = 0; std::string err;
    explicit JsonParser(const std::string& t) : s(t) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
    bool fail(const char* m) { if (err.empty()) err = std::string(m) + " at offset " + std::to_string(i); return false; }
    bool str(std::string& out) {
        if (s[i] != '"') return fail("expected string"); i++;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c == '\\' && i < s.size()) {
                char e = s[i++];
                if (e == 'n') out += '\n'; else if (e == 't') out += '\t'; else if (e == 'r') out += '\r'; else if (e == 'b') out += '\b'; else if (e == 'f') out += '\f';
                else if (e == 'u' && i + 4 <= s.size()) { unsigned cp = (unsigned)strtoul(s.substr(i, 4).c_str(), nullptr, 16); i += 4; if (cp < 0x80) out += (char)cp; else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 63)); } else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63)); } }
                else out += e;
            } else out += c;
        }
        if (i >= s.size()) return fail("unterminated string"); i++; return true;
    }
    bool val(JsonValue& v, int depth) {
        if (depth > 200) return fail("nesting too deep");
        ws(); if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            v.type = JsonValue::Object; i++; ws();
            if (i < s.size() && s[i] == '}') { i++; return true; }
            while (true) {
                ws(); std::pair<std::string, JsonValue> kv; if (i >= s.size() || !str(kv.first)) return fail("expected key");
                ws(); if (i >= s.size() || s[i] != ':') return fail("expected ':'"); i++;
                if (!val(kv.second, depth + 1)) return false; v.obj.push_back(std::move(kv));
                ws(); if (i < s.size() && s[i] == ',') { i++; continue; } if (i < s.size() && s[i] == '}') { i++; return true; } return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = JsonValue::Array; i++; ws();
            if (i < s.size() && s[i] == ']') { i++; return true; }
            while (true) {
                JsonValue e; if (!val(e, depth + 1)) return false; v.arr.push_back(std::move(e));
                ws(); if (i < s.size() && s[i] == ',') { i++; continue; } if (i < s.size() && s[i] == ']') { i++; return true; } return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = JsonValue::String; return str(v.str); }
        if (s.compare(i, 4, "true") == 0) { v.type = JsonValue::Bool; v.b = true; i += 4; return true; }
        if (s.compare(i, 5, "false") == 0) { v.type = JsonValue::Bool; v.b = false; i += 5; return true; }
        if (s.compare(i, 4, "null") == 0) { v.type = JsonValue::Null; i += 4; return true; }
        char* e; double x = strtod(s.c_str() + i, &e); if (e == s.c_str() + i) return fail("unexpected character");
        v.type = JsonValue::Number; v.num = x; i = (size_t)(e - s.c_str()); return true;
    }
};
}

bool parseJson(const std::string& text, JsonValue& out, std::string* err) {
    JsonParser p(text); out = JsonValue();
    bool ok = p.val(out, 0); if (ok) { p.ws(); if (p.i != text.size()) ok = p.fail("trailing characters"); }
    if (!ok && err) *err = p.err;
    return ok;
}
