#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "json.hpp"  

using json = nlohmann::json;
using namespace std;


string toLower(const string &s) {
    string temp = s;
    transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
    return temp;
}


string trim(const string &s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == string::npos) ? "" : s.substr(start, end - start + 1);
}


void query(const vector<map<string,string>> &dataset, const string &field, const string &search, vector<map<string,string>> &result) {
    for (auto &entry : dataset) {
        if (entry.count(field) && toLower(trim(entry.at(field))) == toLower(search)) {
            result.push_back(entry);
        }
    }

    if (result.empty())
        cout << "No exact matches found.\n";
    else {
        cout << "Exact matches:\n";
        for (auto &entry : result) {
            for (auto &kv : entry)
                cout << kv.first << ": " << kv.second << ", ";
            cout << endl;
        }
    }
    cout << endl;
}


void similarquery(const vector<map<string,string>> &dataset, const string &field, const string &search, const vector<map<string,string>> &exactResult) {
    cout << "Similar matches:\n";
    bool found = false;

    for (auto &entry : dataset) {
        if (entry.count(field) && toLower(trim(entry.at(field))).find(toLower(search)) != string::npos) {
            if (find(exactResult.begin(), exactResult.end(), entry) == exactResult.end()) {
                found = true;
                for (auto &kv : entry)
                    cout << kv.first << ": " << kv.second << ", ";
                cout << endl;
            }
        }
    }

    if (!found) cout << "No similar matches found.\n";
    cout << endl;
}

int main() {
    string filename = "/Users/ayushbhandari/Desktop/KnitSpace/MOCK_DATA-2.json";
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return 1;
    }

    json j;
    infile >> j;

  
    vector<map<string,string>> dataset;
    for (auto &item : j) {
        map<string,string> entry;
        for (auto it = item.begin(); it != item.end(); ++it) {
            string key = trim(it.key());
            string value;
            if (it.value().is_string())
                value = trim(it.value().get<string>());
            else
                value = to_string(it.value().get<int>());
            entry[key] = value;
        }
        dataset.push_back(entry);
    }

    cout << "Search by field (e.g., model, first_name, currency): ";
    string field;
    cin >> field;

    cout << "Enter search keyword -  ";
    cin.ignore();
    string keyword;
    getline(cin, keyword);

    vector<map<string,string>> result;
    query(dataset, field, keyword, result);
    similarquery(dataset, field, keyword, result);

    return 0;
}
