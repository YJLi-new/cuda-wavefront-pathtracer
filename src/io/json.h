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
