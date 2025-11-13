#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <set>

using namespace std;

using Matrix = vector<vector<int>>;

// --------------------- UTILIDADES ---------------------

Matrix generar_matriz_booleana(int filas, int columnas, double prob_uno, mt19937& rng) {
    bernoulli_distribution dist(prob_uno);
    Matrix mat(filas, vector<int>(columnas));
    for (int i = 0; i < filas; ++i)
        for (int j = 0; j < columnas; ++j)
            mat[i][j] = dist(rng);
    return mat;
}
// // Función principal que determina si q es subfila de p
pair<bool, vector<int>> es_subfila(const vector<int>& q, const vector<int>& p) {
    vector<int> p_mas_q;
    for (size_t j = 0; j < q.size(); ++j) {
        if (q[j] == 1 && p[j] == 0) return {false, {}};
        if (p[j] == 1 && q[j] == 0) p_mas_q.push_back(j);
    }
    return {!p_mas_q.empty(), p_mas_q};
}

pair<Matrix, vector<tuple<int, int, string>>> construir_matriz_basica_con_explicacion(const Matrix& MD) {
    int m = MD.size();
    vector<bool> es_basica(m, true);
    vector<tuple<int, int, string>> eliminaciones;

    for (int i = 0; i < m; ++i) {
        if (!es_basica[i]) continue;
        for (int j = 0; j < m; ++j) {
            if (i == j || !es_basica[j]) continue;

            auto [es_sub, cols] = es_subfila(MD[i], MD[j]);
            if (es_sub) {
                stringstream ss;
                ss << "Es subfila de F" << j << " (Col ";
                for (size_t k = 0; k < cols.size(); ++k) {
                    ss << cols[k];
                    if (k < cols.size() - 1) ss << ", ";
                }
                ss << ": F" << j << " tiene 1 donde F" << i << " tiene 0)";
                eliminaciones.emplace_back(i, j, ss.str());
                es_basica[i] = false;
                break;
            }
        }
    }

    Matrix MB;
    for (int i = 0; i < m; ++i)
        if (es_basica[i])
            MB.push_back(MD[i]);

    return {MB, eliminaciones};
}

double calcular_densidad(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) return 0.0;
    int unos = 0, total = mat.size() * mat[0].size();
    for (const auto& f : mat) for (int v : f) if (v == 1) ++unos;
    return static_cast<double>(unos) / total;
}

void mostrar_matriz_con_filas_numeradas(const Matrix& mat, const string& titulo) {
    cout << "\n>>> " << titulo << "\n";
    int filas = mat.size(), cols = filas > 0 ? mat[0].size() : 0;
    cout << "Dimensiones: " << filas << " filas x " << cols << " columnas\n";

    if (cols == 0) { cout << "\n"; return; }

    cout << "    ";
    for (int j = 0; j < cols; ++j) cout << setw(4) << j;
    cout << "\n    +";
    for (int j = 0; j < cols; ++j) cout << "----";
    cout << "+\n";

    for (int i = 0; i < filas; ++i) {
        cout << "F" << setw(2) << i << " |";
        for (int j = 0; j < cols; ++j)
            cout << setw(4) << mat[i][j];
        cout << " |\n";
    }
    cout << "\n";
}

// Recibe las eliminaciones y calcula es_basica internamente
void mostrar_analisis_subfilas(const vector<tuple<int, int, string>>& eliminaciones, int m) {
    cout << "\n>>> ANÁLISIS DE SUBFILAS\n";
    if (eliminaciones.empty()) {
        cout << "Todas las filas son básicas (ninguna es subfila de otra).\n";
        return;
    }

    // Reconstruir es_basica a partir de eliminaciones
    vector<bool> es_basica(m, true);
    for (const auto& [q, _, __] : eliminaciones)
        es_basica[q] = false;

    cout << "FILA | ¿Es básica? | Razón de eliminación\n";
    cout << string(50, '-') << "\n";

    set<int> mostradas;
    for (const auto& [fila_elim, fila_dom, razon] : eliminaciones) {
        if (mostradas.count(fila_elim)) continue;
        mostradas.insert(fila_elim);
        string estado = es_basica[fila_elim] ? "BÁSICA" : "ELIMINADA";
        cout << "F" << setw(2) << fila_elim << " | " << left << setw(11) << estado << " | " << razon << "\n";
    }

    vector<int> basicas, eliminadas;
    for (int i = 0; i < m; ++i) {
        (es_basica[i] ? basicas : eliminadas).push_back(i);
    }

    cout << "\nFilas básicas finales: ";
    for (size_t i = 0; i < basicas.size(); ++i)
        cout << "F" << basicas[i] << (i < basicas.size()-1 ? ", " : "\n");

    if (!eliminadas.empty()) {
        cout << "Filas eliminadas: ";
        for (size_t i = 0; i < eliminadas.size(); ++i)
            cout << "F" << eliminadas[i] << (i < eliminadas.size()-1 ? ", " : "\n");
    }
}

// --------------------- MAIN ---------------------

int main() {
    cout << "=== EJERCICIO 1: GENERACIÓN DE MATRIZ BOOLEANA Y MATRIZ BÁSICA ===\n\n";

    // --- Filas ---
    int filas;
    while (true) {
        cout << "Ingresa el número de filas (1 a 100): ";
        cin >> filas;
        if (cin.fail()) { cin.clear(); cin.ignore(10000, '\n'); cout << "Error: ingresa un número entero.\n"; }
        else if (filas >= 1 && filas <= 100) break;
        else cout << "Error: debe estar entre 1 y 100.\n";
    }

    // --- Columnas ---
    int columnas;
    while (true) {
        cout << "Ingresa el número de columnas (1 a 10): ";
        cin >> columnas;
        if (cin.fail()) { cin.clear(); cin.ignore(10000, '\n'); cout << "Error: ingresa un número entero.\n"; }
        else if (columnas >= 1 && columnas <= 10) break;
        else cout << "Error: debe estar entre 1 y 10.\n";
    }

    // --- Seed ---
    int seed_val;
    cout << "Seed para reproducibilidad (Enter para aleatorio): ";
    cin.ignore(10000, '\n');
    string seed_input;
    getline(cin, seed_input);
    if (!seed_input.empty() && seed_input.find_first_not_of("0123456789") == string::npos) {
        seed_val = stoi(seed_input);
    } else {
        seed_val = random_device{}();
    }

    mt19937 rng(seed_val);

    // --- Probabilidad automática ---
    uniform_real_distribution<double> dist_prob(0.5, 1.0);
    double prob_uno = dist_prob(rng);

    cout << fixed << setprecision(3);
    cout << "Seed usado: " << seed_val << "\n";
    cout << "Probabilidad de 1 (generada automáticamente): " << prob_uno << "\n\n";

    // --- Generar MD ---
    cout << "Generando Matriz de Diferencias (MD) de " << filas << "x" << columnas << "...\n\n";
    Matrix MD = generar_matriz_booleana(filas, columnas, prob_uno, rng);

    // --- Construir MB ---
    auto [MB, eliminaciones] = construir_matriz_basica_con_explicacion(MD);

    // --- Mostrar ---
    mostrar_matriz_con_filas_numeradas(MD, "MATRIZ DE DIFERENCIAS (MD)");
    mostrar_matriz_con_filas_numeradas(MB, "MATRIZ BÁSICA (MB)");
    mostrar_analisis_subfilas(eliminaciones, filas);

    // --- Densidades ---
    double densidad_MD = calcular_densidad(MD);
    double densidad_MB = calcular_densidad(MB);
    int unos_MD = 0, unos_MB = 0;
    for (const auto& f : MD) for (int v : f) if (v == 1) ++unos_MD;
    for (const auto& f : MB) for (int v : f) if (v == 1) ++unos_MB;

    cout << "\n>>> DENSIDADES\n";
    cout << fixed << setprecision(4);
    cout << "Densidad de MD = " << densidad_MD << " (" << unos_MD << " unos de " << MD.size() * columnas << " entradas)\n";
    cout << "Densidad de MB = " << densidad_MB << " (" << unos_MB << " unos de " << (MB.empty() ? 0 : MB.size() * columnas) << " entradas)\n";

    return 0;
}
