# ==============================================================================
# IDENTITAS KELOMPOK
# ==============================================================================
# Nama Kelompok : Mak Jaenal
# Kelas         : 12 - IF - 06
# Anggota       :
# 1. 103112430182 - 'Aarif Rahmaan Jalaluddin Faqiih 
# 2. 103112430185 - Atha Muyassar 
# 3. 103112430232 - Eikel Prinst Sukatendel 
#
# KETERANGAN SUMBER LIBRARY:
# - Menggunakan murni library standar bawaan Python (Standard Library) 
#   seperti csv, math, os, random, statistics, time, dan pathlib.
# ==============================================================================

# ==============================================================================
# Bagian Faqiih tentang Fondasi, Struktur Data, dan Helper Parameter
# ==============================================================================
# Import library standar bawaan Python
import csv
import math
import os
import random
import statistics
import time
from pathlib import Path

# Memenuhi Ketentuan Pengujian (Data Uji Sama & Minimal 5 Variasi Input)
BASE_SEED = 2026 # Menjamin data random yang dihasilkan selalu sama persis untuk setiap algoritma
SIZES = [10, 50, 100, 200, 500] # 5 variasi ukuran input (n) dengan jarak yang cukup jauh
REPETITIONS = int(os.environ.get("KNAPSACK_REPETITIONS", "10")) # Jumlah iterasi untuk mendapatkan waktu rata-rata
WEIGHT_MIN = 10     # Batas minimum berat barang
WEIGHT_MAX = 100    # Batas maksimum berat barang
PRIORITY_MIN = 10   # Batas minimum profit/prioritas
PRIORITY_MAX = 500  # Batas maksimum profit/prioritas
CAPACITY_RATIO = 0.35 # Rasio dari total berat yang dijadikan kapasitas knapsack

# Penentuan path output untuk hasil uji dan grafik
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output_knapsack"
RESULTS_PATH = OUTPUT_DIRECTORY / "hasil_pengujian_knapsack.csv"
GRAPH_PATH = OUTPUT_DIRECTORY / "grafik_waktu_knapsack.svg"

# Representasi struktur data untuk setiap barang (paket)
class Item:
    __slots__ = ("package_id", "weight", "profit", "index", "ratio")

    def __init__(self, package_id, weight, profit, index):
        self.package_id = package_id
        self.weight = weight
        self.profit = profit
        self.index = index
        self.ratio = profit / weight # Perhitungan rasio (profit per weight) untuk heuristik

# Representasi hasil pengujian algoritma Knapsack
class KnapsackResult:
    __slots__ = ("profit", "weight", "selected_indices", "nodes_expanded")

    def __init__(self, profit, weight, selected_indices, nodes_expanded=0):
        self.profit = profit
        self.weight = weight
        # Mengubah jadi tuple tanpa fungsi sorted() bawaan agar 100% aman dari aturan dosen
        self.selected_indices = tuple(selected_indices) 
        self.nodes_expanded = nodes_expanded # Melacak berapa node yang diekspansi (khusus Branch and Bound)

# Node untuk membentuk state-space tree pada algoritma Branch and Bound
class Node:
    __slots__ = ("level", "profit", "weight", "bound", "selected_mask")

    def __init__(self, level, profit, weight, bound, selected_mask):
        self.level = level          # Kedalaman tree (indeks barang)
        self.profit = profit        # Profit saat ini di node tersebut
        self.weight = weight        # Berat saat ini di node tersebut
        self.bound = bound          # Nilai harapan/potensi (Upper Bound)
        self.selected_mask = selected_mask # Bitmask untuk menyimpan status barang terpilih

# Implementasi struktur data Priority Queue (Max-Heap) manual 
# Digunakan dalam Branch and Bound untuk mengeksplorasi node dengan Bound tertinggi
class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, node):
        # Menambahkan node baru di akhir array heap, kemudian melakukan proses "bubble-up"
        self.heap.append(node)
        index = len(self.heap) - 1
        while index > 0:
            parent = (index - 1) // 2
            # Jika bound parent sudah lebih besar, posisi sudah benar
            if self.heap[parent].bound >= self.heap[index].bound:
                break
            # Jika tidak, tukar dengan parent-nya
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent

    def pop(self):
        if not self.heap:
            return None
        root = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            # Pindahkan elemen terakhir ke akar, lalu lakukan "bubble-down"
            self.heap[0] = last
            index = 0
            while True:
                left = 2 * index + 1
                right = left + 1
                largest = index
                
                # Cek anak kiri
                if left < len(self.heap) and self.heap[left].bound > self.heap[largest].bound:
                    largest = left
                # Cek anak kanan
                if right < len(self.heap) and self.heap[right].bound > self.heap[largest].bound:
                    largest = right
                # Jika properti heap sudah terpenuhi
                if largest == index:
                    break
                # Tukar dengan anak yang lebih besar
                self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
                index = largest
        return root

    def is_empty(self):
        return not self.heap

# Fungsi pembentuk dataset (paket) dengan sifat deterministik berkat Random Seed
def generate_items(count):
    generator = random.Random(BASE_SEED)
    return [
        Item(
            f"PKT-{index + 1:04d}",
            generator.randint(WEIGHT_MIN, WEIGHT_MAX),
            generator.randint(PRIORITY_MIN, PRIORITY_MAX),
            index,
        )
        for index in range(count)
    ]

# Mengompilasi status akhir dari barang-barang yang dipilih
def build_result(items, selected_indices, nodes_expanded=0):
    selected_set = set(selected_indices)
    profit = sum(item.profit for item in items if item.index in selected_set)
    weight = sum(item.weight for item in items if item.index in selected_set)
    return KnapsackResult(profit, weight, selected_set, nodes_expanded)

# Fungsi Heuristik Upper Bound untuk algoritma Branch and Bound
def calculate_bound(node, capacity, sorted_items):
    # Jika berat node ini melebihi kapasitas, bound-nya dibatalkan (-1)
    if node.weight > capacity:
        return -1.0
    # Jika berat tepat memenuhi kapasitas, bound adalah profit saat ini
    if node.weight == capacity:
        return float(node.profit)
    
    profit_bound = float(node.profit)
    total_weight = node.weight
    index = node.level + 1
    
    # Tambahkan sisa item utuh selama kapasitas masih muat
    while index < len(sorted_items) and total_weight + sorted_items[index].weight <= capacity:
        total_weight += sorted_items[index].weight
        profit_bound += sorted_items[index].profit
        index += 1
        
    # Jika masih ada sisa ruang, gunakan fraksional bagian dari barang selanjutnya untuk batas (bound)
    if index < len(sorted_items):
        profit_bound += (capacity - total_weight) * sorted_items[index].ratio
    return profit_bound

# Mengubah sekumpulan indeks jadi nilai bitmask untuk efisiensi Branch & Bound
def mask_from_indices(indices):
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask

# ==============================================================================
# Bagian Atha tentang Algoritma Utama (Greedy, DP, Branch and Bound)
# ==============================================================================

# Algoritma sorting manual (Bubble Sort) berdasarkan rasio terbesar (profit/weight)
def sort_items_manual(items):
    arr = items[:] 
    n = len(arr)
    
    for i in range(n):
        for j in range(i + 1, n):
            tukar = False
            
            # Prioritas 1: Rasio paling besar
            if arr[j].ratio > arr[i].ratio:
                tukar = True
            elif arr[j].ratio == arr[i].ratio:
                # Prioritas 2: Jika rasio sama, pilih weight terkecil
                if arr[j].weight < arr[i].weight:
                    tukar = True
                elif arr[j].weight == arr[i].weight:
                    # Prioritas 3: Jika weight sama, pilih profit terbesar
                    if arr[j].profit > arr[i].profit:
                        tukar = True
                    elif arr[j].profit == arr[i].profit:
                        # Prioritas 4: Berdasarkan indeks (mempertahankan stabilitas)
                        if arr[j].index < arr[i].index:
                            tukar = True
            
            # Menukar elemen
            if tukar:
                arr[i], arr[j] = arr[j], arr[i]
                
    return arr


# Algoritma 1: Greedy berdasar Density/Rasio (Pendekatan Heuristik)
# Referensi: Algoritma ini mengambil referensi dari konsep Fractional Greedy Heuristic
# pada buku "Introduction to Algorithms, 4th Edition" oleh Cormen dkk. (MIT Press, 2022).
def greedy_knapsack(items, capacity):
    # Urutkan item berdasarkan urutan prioritas rasionya
    sorted_items = sort_items_manual(items) 
    selected_indices = []
    total_weight = 0
    total_profit = 0
    
    # Ambil barang selama masih muat di dalam kapasitas
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected_indices.append(item.index)
            total_weight += item.weight
            total_profit += item.profit
    return KnapsackResult(total_profit, total_weight, selected_indices)


# Algoritma 2: Dynamic Programming (Bottom-up 1D Array)
# Referensi: Algoritma ini mengambil referensi dari optimalisasi memori untuk 0/1 Knapsack 
# dari buku "Algorithm Design" oleh Kleinberg & Tardos (Cornell University).
def dp_knapsack(items, capacity):
    dp = [0] * (capacity + 1) # Array untuk menyimpan profit maksimal setiap tingkat kapasitas
    decisions = [] # Menyimpan riwayat keputusan pengambilan item
    
    # Mulai evaluasi per item
    for item in items:
        row = bytearray(capacity + 1)
        # Evaluasi dari kapasitas teratas menuju batas berat item untuk mencegah penghitungan ganda
        for current_capacity in range(capacity, item.weight - 1, -1):
            candidate = dp[current_capacity - item.weight] + item.profit
            # Jika profit jika mengambil item lebih besar dari sebelumnya, ambil item tersebut
            if candidate > dp[current_capacity]:
                dp[current_capacity] = candidate
                row[current_capacity] = 1 
        decisions.append(row)
        
    # Proses backtracking (rekonstruksi mundur) untuk mencari item mana yang terpilih
    selected_indices = []
    current_capacity = capacity
    for index in range(len(items) - 1, -1, -1):
        # Jika ada keputusan pengambilan dari history di indeks tersebut
        if decisions[index][current_capacity]:
            selected_indices.append(items[index].index)
            current_capacity -= items[index].weight # kurangi kapasitas saat ini dengan berat barang
            
    result = build_result(items, selected_indices)
    
    # Validasi konsistensi nilai DP dan proses hasil rekonstuksi
    if result.profit != dp[capacity]:
        raise RuntimeError("Rekonstruksi solusi Dynamic Programming tidak konsisten.")
    return result


# Algoritma 3: Branch and Bound dengan Max-Heap Priority Queue
# Referensi: Algoritma ini mengambil referensi perancangan State Space Tree dan Upper Bound
# dari jurnal klasik oleh Martello & Toth serta buku "Foundations of Algorithms" (Neapolitan).
def branch_and_bound_knapsack(items, capacity):
    sorted_items = sort_items_manual(items) 
    
    # Inisialisasi batas bawah global (best profit) pakai solusi Greedy untuk pangkas node
    initial = greedy_knapsack(items, capacity)
    best_profit = initial.profit
    best_weight = initial.weight
    best_mask = mask_from_indices(initial.selected_indices)
    
    queue = MaxHeap()
    # Inisiasi Root/Akar pohon
    root = Node(-1, 0, 0, 0.0, 0)
    root.bound = calculate_bound(root, capacity, sorted_items)
    queue.push(root)
    nodes_expanded = 0
    
    # Terus evaluasi node yang ada dalam queue (Best-First Search)
    while not queue.is_empty():
        current = queue.pop()
        
        # Jika nilai bound node saat ini lebih kecil atau sama dengan solusi yang sudah kita punya, buang (prune)
        if current.bound <= best_profit:
            continue
            
        nodes_expanded += 1
        next_level = current.level + 1
        
        # Jika sudah sampai batas kedalaman item, skip
        if next_level >= len(sorted_items):
            continue
            
        item = sorted_items[next_level]
        
        # KASUS 1: Simulasi mengambil (take) item tersebut
        take_weight = current.weight + item.weight
        take_profit = current.profit + item.profit
        take_mask = current.selected_mask | (1 << item.index)
        
        # Jika bobot memenuhi dan profit melebih current best, jadikan best profit baru
        if take_weight <= capacity and take_profit > best_profit:
            best_profit = take_profit
            best_weight = take_weight
            best_mask = take_mask
            
        # Bentuk node untuk kasus "take" dan hitung Bound
        take_node = Node(next_level, take_profit, take_weight, 0.0, take_mask)
        take_node.bound = calculate_bound(take_node, capacity, sorted_items)
        # Tambahkan ke queue hanya jika menjanjikan
        if take_node.bound > best_profit:
            queue.push(take_node)
            
        # KASUS 2: Simulasi melewati (skip) item tersebut
        skip_node = Node(next_level, current.profit, current.weight, 0.0, current.selected_mask)
        skip_node.bound = calculate_bound(skip_node, capacity, sorted_items)
        # Tambahkan ke queue jika node ini berpotensi memberikan profit lebih tinggi
        if skip_node.bound > best_profit:
            queue.push(skip_node)
            
    # Menerjemahkan bitmask terakhir menjadi array indeks barang-barang yang dipilih
    selected_indices = [item.index for item in items if best_mask & (1 << item.index)]
    result = build_result(items, selected_indices, nodes_expanded)
    
    # Validasi error
    if result.profit != best_profit or result.weight != best_weight:
        raise RuntimeError("Rekonstruksi solusi Branch and Bound tidak konsisten.")
    return result

# ==============================================================================
# Bagian Eikel tentang Pengujian, Validasi, Visualisasi (CSV/SVG), dan Eksekusi
# ==============================================================================

# Melakukan sanity check: apakah indeks duplikat? profit/berat sesuai?, over capacity?
def validate_result(items, capacity, result):
    if len(result.selected_indices) != len(set(result.selected_indices)):
        raise RuntimeError("Solusi memuat paket yang sama lebih dari satu kali.")
    rebuilt = build_result(items, result.selected_indices, result.nodes_expanded)
    if rebuilt.profit != result.profit or rebuilt.weight != result.weight:
        raise RuntimeError("Profit atau berat solusi tidak konsisten.")
    if result.weight > capacity:
        raise RuntimeError("Solusi melebihi kapasitas kendaraan.")

# Melakukan pengujian kecepatan ekskusi (benchmarking) suatu algoritma
def benchmark_algorithm(function, items, capacity, repetitions):
    warmup_result = function(items, capacity) # Pengujian pemanasan agar memori stabil
    validate_result(items, capacity, warmup_result)
    expected_profit = warmup_result.profit
    elapsed_times = []
    latest_result = warmup_result
    
    # Pengulangan (iterasi) pengujian kecepatan untuk mencari median/rata-rata akurat
    for _ in range(repetitions):
        start = time.perf_counter()
        latest_result = function(items, capacity)
        elapsed_times.append(time.perf_counter() - start)
        validate_result(items, capacity, latest_result)
        # Pastikan algoritma konsisten meskipun diulang berkali-kali
        if latest_result.profit != expected_profit:
            raise RuntimeError("Algoritma menghasilkan profit yang tidak konsisten.")
            
    return {
        "result": latest_result,
        "median_seconds": statistics.median(elapsed_times),
        "average_seconds": statistics.fmean(elapsed_times),
        "minimum_seconds": min(elapsed_times),
    }

# Fungs utama yang mengoordinasikan loop pengujian di berbagai dataset (N)
def run_experiments():
    all_items = generate_items(max(SIZES))
    algorithms = [
        ("Greedy Density", greedy_knapsack),
        ("Dynamic Programming", dp_knapsack),
        ("Branch and Bound", branch_and_bound_knapsack),
    ]
    records = []
    
    # Pengujian setiap variasi ukuran data N
    for size in SIZES:
        items = all_items[:size]
        capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
        size_records = []
        
        # Pengujian algoritma satu persatu di dalam ukuran data saat ini
        for algorithm_name, function in algorithms:
            benchmark = benchmark_algorithm(function, items, capacity, REPETITIONS)
            result = benchmark["result"]
            item_by_index = {item.index: item for item in items}
            selected_ids = [item_by_index[index].package_id for index in result.selected_indices]
            
            # Catat metriks ke dictionary list
            record = {
                "n": size,
                "capacity": capacity,
                "algorithm": algorithm_name,
                "median_ms": benchmark["median_seconds"] * 1000,
                "average_ms": benchmark["average_seconds"] * 1000,
                "minimum_ms": benchmark["minimum_seconds"] * 1000,
                "profit": result.profit,
                "weight": result.weight,
                "selected_count": len(result.selected_indices),
                "selected_ids": selected_ids,
                "nodes_expanded": result.nodes_expanded,
            }
            size_records.append(record)
            
        # Validasi Optimality Gap, Pastikan DP & Branch Bound membuahkan profit maksimal yang sama
        dp_profit = next(record["profit"] for record in size_records if record["algorithm"] == "Dynamic Programming")
        bb_profit = next(record["profit"] for record in size_records if record["algorithm"] == "Branch and Bound")
        if dp_profit != bb_profit:
            raise RuntimeError(f"DP dan Branch and Bound berbeda pada n={size}.")
            
        # Hitung selisih dari Greedy dengan algoritma optimum (DP/Branch & Bound)
        for record in size_records:
            gap = dp_profit - record["profit"]
            record["optimality_gap"] = gap
            record["gap_percent"] = 0.0 if dp_profit == 0 else gap / dp_profit * 100
        records.extend(size_records)
    return records

# Menulis rekaman pengujian ke dalam file .CSV
def write_results_csv(records):
    fieldnames = [
        "n", "capacity", "algorithm", "median_ms", "average_ms", "minimum_ms",
        "profit", "weight", "selected_count", "optimality_gap", "gap_percent",
        "nodes_expanded", "selected_ids",
    ]
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["median_ms"] = f"{record['median_ms']:.9f}"
            row["average_ms"] = f"{record['average_ms']:.9f}"
            row["minimum_ms"] = f"{record['minimum_ms']:.9f}"
            row["gap_percent"] = f"{record['gap_percent']:.6f}"
            row["selected_ids"] = ";".join(record["selected_ids"])
            writer.writerow(row)

# Fungsi helper membuat tag text element untuk rendering SVG
def svg_text(x, y, text, size=14, anchor="middle", weight="normal", fill="rgb(30, 41, 59)", rotation=None):
    transform = f' transform="rotate({rotation} {x} {y})"' if rotation is not None else ""
    safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{safe_text}</text>'

# Fungsi pembuat file Grafik Garis (SVG) murni tanpa library plot eskternal (Matplotlib)
def write_graph_svg(records):
    width = 1200
    height = 760
    left = 110
    right = 60
    top = 90
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    
    # Warna perwakilan masing-masing algoritma
    colors = {
        "Greedy Density": "rgb(37, 99, 235)",
        "Dynamic Programming": "rgb(220, 38, 38)",
        "Branch and Bound": "rgb(5, 150, 105)",
    }
    
    # Perhitungan skala Logaritmik dasar
    values = [max(record["median_ms"], 0.000001) for record in records]
    y_min = 10 ** math.floor(math.log10(min(values)))
    y_max = 10 ** math.ceil(math.log10(max(values)))
    if y_min == y_max:
        y_max = y_min * 10
    x_min = math.log10(min(SIZES))
    x_max = math.log10(max(SIZES))

    def x_position(size):
        return left + (math.log10(size) - x_min) / (x_max - x_min) * plot_width

    def y_position(value):
        normalized = (math.log10(max(value, 0.000001)) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        return top + plot_height - normalized * plot_height

    # Rendering area SVG Utama
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgb(248, 250, 252)"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="rgb(203, 213, 225)"/>',
        svg_text(width / 2, 38, "Perbandingan Median Waktu Eksekusi 0/1 Knapsack", 23, weight="bold"),
        svg_text(width / 2, 64, f"{REPETITIONS} pengulangan per algoritma, skala waktu logaritmik", 14, fill="rgb(71, 85, 105)"),
    ]
    
    # Rendering Grid Horisontal SVG
    lower_power = int(math.floor(math.log10(y_min)))
    upper_power = int(math.ceil(math.log10(y_max)))
    for power in range(lower_power, upper_power + 1):
        value = 10 ** power
        y = y_position(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="rgb(226, 232, 240)"/>')
        lines.append(svg_text(left - 12, y + 5, f"{value:g}", 12, anchor="end"))
        
    # Rendering Grid Vertikal SVG
    for size in SIZES:
        x = x_position(size)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="rgb(241, 245, 249)"/>')
        lines.append(svg_text(x, top + plot_height + 28, size, 13))
        
    lines.append(svg_text(width / 2, height - 28, "Jumlah paket kandidat (n)", 15, weight="bold"))
    lines.append(svg_text(28, top + plot_height / 2, "Median waktu (ms)", 15, weight="bold", rotation=-90))
    
    # Rendering titik (node) dan garis (path) Algoritma di Grafik
    for algorithm, color in colors.items():
        algorithm_records = sorted(
            (record for record in records if record["algorithm"] == algorithm),
            key=lambda record: record["n"],
        )
        points = " ".join(
            f"{x_position(record['n']):.2f},{y_position(record['median_ms']):.2f}"
            for record in algorithm_records
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for record in algorithm_records:
            x = x_position(record["n"])
            y = y_position(record["median_ms"])
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="white" stroke-width="2"/>')
            
    # Rendering Legend/Keterangan Warna
    legend_x = left + 20
    legend_y = top + 24
    for index, (algorithm, color) in enumerate(colors.items()):
        y = legend_y + index * 25
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        lines.append(svg_text(legend_x + 38, y + 5, algorithm, 13, anchor="start"))
        
    lines.append("</svg>")
    GRAPH_PATH.write_text("\n".join(lines), encoding="utf-8")

# Fungsi membatasi print out id paket agar CLI tidak terlalu panjang
def selected_preview(selected_ids, limit=10):
    if len(selected_ids) <= limit:
        return ", ".join(selected_ids)
    return ", ".join(selected_ids[:limit]) + f", dan {len(selected_ids) - limit} paket lainnya"

# Membentuk teks grid ASCII untuk print di terminal
def result_table_lines(records):
    header = f"{'n':>5} | {'Kapasitas':>9} | {'Algoritma':<19} | {'Median ms':>12} | {'Profit':>8} | {'Berat':>7} | {'Paket':>5} | {'Gap':>6}"
    separator = "-" * len(header)
    lines = [header, separator]
    for record in records:
        lines.append(
            f"{record['n']:>5} | {record['capacity']:>9} | {record['algorithm']:<19} | "
            f"{record['median_ms']:>12.6f} | {record['profit']:>8} | {record['weight']:>7} | "
            f"{record['selected_count']:>5} | {record['optimality_gap']:>6}"
        )
    return lines

# Menampilkan seluruh hasil pengujian dan letak file output kepada user
def print_results(records):
    print()
    for line in result_table_lines(records):
        print(line)
    print()
    for record in records:
        print(
            f"n={record['n']} | {record['algorithm']} | paket terpilih: "
            f"{selected_preview(record['selected_ids'])}"
        )
    print()
    print(f"CSV: {RESULTS_PATH}")
    print(f"Grafik: {GRAPH_PATH}")

# Point of Entry, menjalankan flow utama saat file Python dijalankan
def main():
    if REPETITIONS < 2:
        raise ValueError("Jumlah pengulangan minimal 2.")
    
    # Jalankan eksperimen pengujian
    records = run_experiments()
    
    # Siapkan direktori untuk file output (CSV dan SVG)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    # Generate keluaran (Export)
    write_results_csv(records)
    write_graph_svg(records)
    print_results(records)

# Mencegah otomatis tertrigger jika di-import ke file Python lain
if __name__ == "__main__":
    main()

# ==============================================================================
# DAFTAR PUSTAKA & REFERENSI ALGORITMA
# ==============================================================================
# 1. Pendekatan Greedy & Dynamic Programming:
#    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). 
#    "Introduction to Algorithms" (4th ed.). MIT Press. 
#    (Bab 15 & 16 membahas penyelesaian optimal untuk masalah optimasi Knapsack).
#
# 2. Desain Algoritma dan Optimalisasi Memori:
#    Kleinberg, J., & Tardos, E. (2006). "Algorithm Design". Pearson Education. 
#    (Digunakan sebagai referensi pembentukan Bottom-Up DP 1D Array).
#
# 3. Branch and Bound & Upper Bound Knapsack:
#    Neapolitan, R. E. (2014). "Foundations of Algorithms" (5th ed.). 
#    Jones & Bartlett Learning. (Bab 6: Best-First Search dengan Priority Queue).
#
# 4. Riset Fundamental Bounding Knapsack:
#    Martello, S., & Toth, P. (1990). "Knapsack Problems: Algorithms and Computer 
#    Implementations". John Wiley & Sons.
# ==============================================================================
