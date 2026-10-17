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
