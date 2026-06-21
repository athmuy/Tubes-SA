# ==============================================================================
# Bagian Faqiih tentang Fondasi, Struktur Data, dan Helper Parameter
# ==============================================================================
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
REPETITIONS = int(os.environ.get("KNAPSACK_REPETITIONS", "10"))
WEIGHT_MIN = 10
WEIGHT_MAX = 100
PRIORITY_MIN = 10
PRIORITY_MAX = 500
CAPACITY_RATIO = 0.35
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output_knapsack"
RESULTS_PATH = OUTPUT_DIRECTORY / "hasil_pengujian_knapsack.csv"
GRAPH_PATH = OUTPUT_DIRECTORY / "grafik_waktu_knapsack.svg"


class Item:
    __slots__ = ("package_id", "weight", "profit", "index", "ratio")

    def __init__(self, package_id, weight, profit, index):
        self.package_id = package_id
        self.weight = weight
        self.profit = profit
        self.index = index
        self.ratio = profit / weight


class KnapsackResult:
    __slots__ = ("profit", "weight", "selected_indices", "nodes_expanded")

    def __init__(self, profit, weight, selected_indices, nodes_expanded=0):
        self.profit = profit
        self.weight = weight
        # Mengubah jadi tuple tanpa fungsi sorted() bawaan agar 100% aman dari aturan dosen
        self.selected_indices = tuple(selected_indices) 
        self.nodes_expanded = nodes_expanded


class Node:
    __slots__ = ("level", "profit", "weight", "bound", "selected_mask")

    def __init__(self, level, profit, weight, bound, selected_mask):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.bound = bound
        self.selected_mask = selected_mask


class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, node):
        self.heap.append(node)
        index = len(self.heap) - 1
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[parent].bound >= self.heap[index].bound:
                break
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent

    def pop(self):
        if not self.heap:
            return None
        root = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            index = 0
            while True:
                left = 2 * index + 1
                right = left + 1
                largest = index
                if left < len(self.heap) and self.heap[left].bound > self.heap[largest].bound:
                    largest = left
                if right < len(self.heap) and self.heap[right].bound > self.heap[largest].bound:
                    largest = right
                if largest == index:
                    break
                self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
                index = largest
        return root

    def is_empty(self):
        return not self.heap


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


def build_result(items, selected_indices, nodes_expanded=0):
    selected_set = set(selected_indices)
    profit = sum(item.profit for item in items if item.index in selected_set)
    weight = sum(item.weight for item in items if item.index in selected_set)
    return KnapsackResult(profit, weight, selected_set, nodes_expanded)


def calculate_bound(node, capacity, sorted_items):
    if node.weight > capacity:
        return -1.0
    if node.weight == capacity:
        return float(node.profit)
    profit_bound = float(node.profit)
    total_weight = node.weight
    index = node.level + 1
    while index < len(sorted_items) and total_weight + sorted_items[index].weight <= capacity:
        total_weight += sorted_items[index].weight
        profit_bound += sorted_items[index].profit
        index += 1
    if index < len(sorted_items):
        profit_bound += (capacity - total_weight) * sorted_items[index].ratio
    return profit_bound


def mask_from_indices(indices):
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask

# ==============================================================================
# Bagian Atha tentang Algoritma Utama (Greedy, DP, Branch and Bound)
# Memenuhi Ketentuan Minimum (Menyelesaikan studi kasus > 2 strategi)
# Memenuhi syarat dosen (algoritma dibuat "from scratch", dilarang pakai library sort)
# ==============================================================================

# Fungsi helper pengurutan manual "From Scratch" tanpa menggunakan library bawaan
def sort_items_manual(items):
    arr = items[:] 
    n = len(arr)
    
    for i in range(n):
        for j in range(i + 1, n):
            tukar = False
            
            # Prioritas 1: Rasio profit/berat lebih besar ditaruh di depan
            if arr[j].ratio > arr[i].ratio:
                tukar = True
            elif arr[j].ratio == arr[i].ratio:
                # Prioritas 2: Jika rasio sama, pilih yang beratnya paling ringan
                if arr[j].weight < arr[i].weight:
                    tukar = True
                elif arr[j].weight == arr[i].weight:
                    # Prioritas 3: Jika berat sama, pilih yang profitnya paling besar
                    if arr[j].profit > arr[i].profit:
                        tukar = True
                    elif arr[j].profit == arr[i].profit:
                        # Prioritas 4: Jika kembar identik, urutkan berdasarkan index
                        if arr[j].index < arr[i].index:
                            tukar = True
            
            # Tukar posisi
            if tukar:
                arr[i], arr[j] = arr[j], arr[i]
                
    return arr


def greedy_knapsack(items, capacity):
    sorted_items = sort_items_manual(items) # Menggunakan pengurutan manual
    selected_indices = []
    total_weight = 0
    total_profit = 0
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected_indices.append(item.index)
            total_weight += item.weight
            total_profit += item.profit
    return KnapsackResult(total_profit, total_weight, selected_indices)


def dp_knapsack(items, capacity):
    dp = [0] * (capacity + 1)
    decisions = []
    for item in items:
        row = bytearray(capacity + 1)
        for current_capacity in range(capacity, item.weight - 1, -1):
            candidate = dp[current_capacity - item.weight] + item.profit
            if candidate > dp[current_capacity]:
                dp[current_capacity] = candidate
                row[current_capacity] = 1 
        decisions.append(row)
        
    selected_indices = []
    current_capacity = capacity
    for index in range(len(items) - 1, -1, -1):
        if decisions[index][current_capacity]:
            selected_indices.append(items[index].index)
            current_capacity -= items[index].weight
            
    result = build_result(items, selected_indices)
    if result.profit != dp[capacity]:
        raise RuntimeError("Rekonstruksi solusi Dynamic Programming tidak konsisten.")
    return result


def branch_and_bound_knapsack(items, capacity):
    sorted_items = sort_items_manual(items) # Menggunakan pengurutan manual
    
    initial = greedy_knapsack(items, capacity)
    best_profit = initial.profit
    best_weight = initial.weight
    best_mask = mask_from_indices(initial.selected_indices)
    
    queue = MaxHeap()
    root = Node(-1, 0, 0, 0.0, 0)
    root.bound = calculate_bound(root, capacity, sorted_items)
    queue.push(root)
    nodes_expanded = 0
    
    while not queue.is_empty():
        current = queue.pop()
        
        if current.bound <= best_profit:
            continue
            
        nodes_expanded += 1
        next_level = current.level + 1
        if next_level >= len(sorted_items):
            continue
            
        item = sorted_items[next_level]
        
        take_weight = current.weight + item.weight
        take_profit = current.profit + item.profit
        take_mask = current.selected_mask | (1 << item.index)
        
        if take_weight <= capacity and take_profit > best_profit:
            best_profit = take_profit
            best_weight = take_weight
            best_mask = take_mask
            
        take_node = Node(next_level, take_profit, take_weight, 0.0, take_mask)
        take_node.bound = calculate_bound(take_node, capacity, sorted_items)
        if take_node.bound > best_profit:
            queue.push(take_node)
            
        skip_node = Node(next_level, current.profit, current.weight, 0.0, current.selected_mask)
        skip_node.bound = calculate_bound(skip_node, capacity, sorted_items)
        if skip_node.bound > best_profit:
            queue.push(skip_node)
            
    selected_indices = [item.index for item in items if best_mask & (1 << item.index)]
    result = build_result(items, selected_indices, nodes_expanded)
    
    if result.profit != best_profit or result.weight != best_weight:
        raise RuntimeError("Rekonstruksi solusi Branch and Bound tidak konsisten.")
    return result

# ==============================================================================
# Bagian Eikel tentang Pengujian, Validasi, Visualisasi (CSV/SVG), dan Eksekusi
# Memenuhi ketentuan: Catat running time, diagram SVG otomatis
# ==============================================================================

def validate_result(items, capacity, result):
    if len(result.selected_indices) != len(set(result.selected_indices)):
        raise RuntimeError("Solusi memuat paket yang sama lebih dari satu kali.")
    rebuilt = build_result(items, result.selected_indices, result.nodes_expanded)
    if rebuilt.profit != result.profit or rebuilt.weight != result.weight:
        raise RuntimeError("Profit atau berat solusi tidak konsisten.")
    if result.weight > capacity:
        raise RuntimeError("Solusi melebihi kapasitas kendaraan.")


def benchmark_algorithm(function, items, capacity, repetitions):
    warmup_result = function(items, capacity)
    validate_result(items, capacity, warmup_result)
    expected_profit = warmup_result.profit
    elapsed_times = []
    latest_result = warmup_result
    for _ in range(repetitions):
        start = time.perf_counter()
        latest_result = function(items, capacity)
        elapsed_times.append(time.perf_counter() - start)
        validate_result(items, capacity, latest_result)
        if latest_result.profit != expected_profit:
            raise RuntimeError("Algoritma menghasilkan profit yang tidak konsisten.")
    return {
        "result": latest_result,
        "median_seconds": statistics.median(elapsed_times),
        "average_seconds": statistics.fmean(elapsed_times),
        "minimum_seconds": min(elapsed_times),
    }


def run_experiments():
    all_items = generate_items(max(SIZES))
    algorithms = [
        ("Greedy Density", greedy_knapsack),
        ("Dynamic Programming", dp_knapsack),
        ("Branch and Bound", branch_and_bound_knapsack),
    ]
    records = []
    for size in SIZES:
        items = all_items[:size]
        capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
        size_records = []
        for algorithm_name, function in algorithms:
            benchmark = benchmark_algorithm(function, items, capacity, REPETITIONS)
            result = benchmark["result"]
            item_by_index = {item.index: item for item in items}
            selected_ids = [item_by_index[index].package_id for index in result.selected_indices]
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
        dp_profit = next(record["profit"] for record in size_records if record["algorithm"] == "Dynamic Programming")
        bb_profit = next(record["profit"] for record in size_records if record["algorithm"] == "Branch and Bound")
        if dp_profit != bb_profit:
            raise RuntimeError(f"DP dan Branch and Bound berbeda pada n={size}.")
        for record in size_records:
            gap = dp_profit - record["profit"]
            record["optimality_gap"] = gap
            record["gap_percent"] = 0.0 if dp_profit == 0 else gap / dp_profit * 100
        records.extend(size_records)
    return records


def write_results_csv(records):
    fieldnames = [
        "n",
        "capacity",
        "algorithm",
        "median_ms",
        "average_ms",
        "minimum_ms",
        "profit",
        "weight",
        "selected_count",
        "optimality_gap",
        "gap_percent",
        "nodes_expanded",
        "selected_ids",
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


def svg_text(x, y, text, size=14, anchor="middle", weight="normal", fill="rgb(30, 41, 59)", rotation=None):
    transform = f' transform="rotate({rotation} {x} {y})"' if rotation is not None else ""
    safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{safe_text}</text>'


def write_graph_svg(records):
    width = 1200
    height = 760
    left = 110
    right = 60
    top = 90
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = {
        "Greedy Density": "rgb(37, 99, 235)",
        "Dynamic Programming": "rgb(220, 38, 38)",
        "Branch and Bound": "rgb(5, 150, 105)",
    }
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

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgb(248, 250, 252)"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="rgb(203, 213, 225)"/>',
        svg_text(width / 2, 38, "Perbandingan Median Waktu Eksekusi 0/1 Knapsack", 23, weight="bold"),
        svg_text(width / 2, 64, f"{REPETITIONS} pengulangan per algoritma, skala waktu logaritmik", 14, fill="rgb(71, 85, 105)"),
    ]
    lower_power = int(math.floor(math.log10(y_min)))
    upper_power = int(math.ceil(math.log10(y_max)))
    for power in range(lower_power, upper_power + 1):
        value = 10 ** power
        y = y_position(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="rgb(226, 232, 240)"/>')
        lines.append(svg_text(left - 12, y + 5, f"{value:g}", 12, anchor="end"))
    for size in SIZES:
        x = x_position(size)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="rgb(241, 245, 249)"/>')
        lines.append(svg_text(x, top + plot_height + 28, size, 13))
    lines.append(svg_text(width / 2, height - 28, "Jumlah paket kandidat (n)", 15, weight="bold"))
    lines.append(svg_text(28, top + plot_height / 2, "Median waktu (ms)", 15, weight="bold", rotation=-90))
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
    legend_x = left + 20
    legend_y = top + 24
    for index, (algorithm, color) in enumerate(colors.items()):
        y = legend_y + index * 25
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        lines.append(svg_text(legend_x + 38, y + 5, algorithm, 13, anchor="start"))
    lines.append("</svg>")
    GRAPH_PATH.write_text("\n".join(lines), encoding="utf-8")


def selected_preview(selected_ids, limit=10):
    if len(selected_ids) <= limit:
        return ", ".join(selected_ids)
    return ", ".join(selected_ids[:limit]) + f", dan {len(selected_ids) - limit} paket lainnya"


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


def main():
    if REPETITIONS < 2:
        raise ValueError("Jumlah pengulangan minimal 2.")
    records = run_experiments()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_results_csv(records)
    write_graph_svg(records)
    print_results(records)


if __name__ == "__main__":
    main()
