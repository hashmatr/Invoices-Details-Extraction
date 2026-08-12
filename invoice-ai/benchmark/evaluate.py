import json
import os
import time
from typing import Dict, Any, List, Tuple
from graph.graph import build_graph
from models.metrics import ProcessingMetrics
from utils.formatting import Normalizer
from decimal import Decimal
from config.settings import settings

class BenchmarkMetrics:
    def __init__(self, name: str):
        self.name = name
        self.total_invoices = 0
        
        self.total_time_ms = 0
        self.ocr_time_ms = 0
        self.structure_time_ms = 0
        self.extraction_time_ms = 0
        self.gemini_time_ms = 0
        self.gemini_calls = 0
        
        self.field_accuracy = {
            "invoice_number": [0, 0],
            "invoice_date": [0, 0],
            "supplier_name": [0, 0],
            "supplier_vat": [0, 0],
            "customer_name": [0, 0],
            "customer_vat": [0, 0],
            "amount_before_vat": [0, 0],
            "vat_amount": [0, 0],
            "total_amount_after_vat": [0, 0],
            "qr_code_present": [0, 0]
        }
        
        self.line_item_metrics = {
            "row_count_match": [0, 0],
            "description": [0, 0],
            "quantity": [0, 0],
            "unit": [0, 0],
            "unit_price": [0, 0],
            "vat_rate": [0, 0],
            "vat_amount": [0, 0],
            "amount": [0, 0],
            "formula_correct": [0, 0]
        }
        
        self.validation_failure_count = 0

def _compare_text(extracted: str, expected: str) -> bool:
    if expected is None:
        return extracted is None
    if extracted is None:
        return False
    return Normalizer.normalize_chinese_text(extracted) == Normalizer.normalize_chinese_text(expected)

def _compare_currency(extracted, expected) -> bool:
    if expected is None:
        return extracted is None
    if extracted is None:
        return False
    
    ext_dec = Normalizer.normalize_currency(extracted)
    exp_dec = Normalizer.normalize_currency(expected)
    
    if ext_dec is None or exp_dec is None:
        return False
        
    return abs(ext_dec - exp_dec) <= Decimal(str(settings.ROUNDING_TOLERANCE))

def _compare_date(extracted, expected) -> bool:
    if expected is None:
        return extracted is None
    if extracted is None:
        return False
        
    ext_date = Normalizer.normalize_date(extracted)
    exp_date = Normalizer.normalize_date(expected)
    return ext_date == exp_date
    
def _update_acc(metrics_dict: dict, field: str, is_correct: bool):
    metrics_dict[field][1] += 1
    if is_correct:
        metrics_dict[field][0] += 1

def run_evaluation(dataset_dir: str, enable_gemini: bool, name: str) -> BenchmarkMetrics:
    app = build_graph(enable_gemini=enable_gemini)
    files = os.listdir(dataset_dir)
    image_files = [f for f in files if not f.endswith("_gt.json")]
    
    metrics = BenchmarkMetrics(name)
    
    for filename in image_files:
        base_name = os.path.splitext(filename)[0]
        gt_filename = f"{base_name}_gt.json"
        file_path = os.path.join(dataset_dir, filename)
        gt_path = os.path.join(dataset_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            continue
            
        metrics.total_invoices += 1
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        initial_state = {
            "file_path": file_path,
            "file_type": "pdf" if filename.lower().endswith(".pdf") else "image",
            "images": [],
            "ocr_data": [],
            "layout_data": {},
            "qr_code_present": False,
            "qr_data": [],
            "invoice_data": None,
            "validation_errors": [],
            "suspicious_fields": [],
            "gemini_corrections": {},
            "metrics": ProcessingMetrics()
        }
        
        start_time = time.time()
        result_state = None
        for event in app.stream(initial_state):
            for _, state in event.items():
                result_state = state
        end_time = time.time()
        
        pm = result_state["metrics"]
        pm.total_time_ms = (end_time - start_time) * 1000
        
        metrics.total_time_ms += pm.total_time_ms
        metrics.ocr_time_ms += pm.ocr_time_ms
        metrics.structure_time_ms += pm.structure_time_ms
        metrics.extraction_time_ms += pm.extraction_time_ms
        metrics.gemini_time_ms += pm.gemini_time_ms
        metrics.gemini_calls += pm.gemini_calls
        
        if pm.final_status == "needs_review":
            metrics.validation_failure_count += 1
            
        invoice = result_state.get("invoice_data")
        
        if not invoice:
            for k in metrics.field_accuracy.keys():
                _update_acc(metrics.field_accuracy, k, False)
            _update_acc(metrics.line_item_metrics, "row_count_match", False)
            continue
            
        _update_acc(metrics.field_accuracy, "invoice_number", _compare_text(invoice.invoice_number.value if invoice.invoice_number else None, gt_data.get("invoice_number")))
        _update_acc(metrics.field_accuracy, "invoice_date", _compare_date(invoice.invoice_date.value if invoice.invoice_date else None, gt_data.get("invoice_date")))
        _update_acc(metrics.field_accuracy, "supplier_name", _compare_text(invoice.supplier.name.value if invoice.supplier and invoice.supplier.name else None, gt_data.get("supplier_name")))
        _update_acc(metrics.field_accuracy, "supplier_vat", _compare_text(invoice.supplier.vat.value if invoice.supplier and invoice.supplier.vat else None, gt_data.get("supplier_vat")))
        _update_acc(metrics.field_accuracy, "customer_name", _compare_text(invoice.customer.name.value if invoice.customer and invoice.customer.name else None, gt_data.get("customer_name")))
        _update_acc(metrics.field_accuracy, "customer_vat", _compare_text(invoice.customer.vat.value if invoice.customer and invoice.customer.vat else None, gt_data.get("customer_vat")))
        
        _update_acc(metrics.field_accuracy, "amount_before_vat", _compare_currency(invoice.amount_before_vat.value if invoice.amount_before_vat else None, gt_data.get("amount_before_vat")))
        _update_acc(metrics.field_accuracy, "vat_amount", _compare_currency(invoice.vat_amount.value if invoice.vat_amount else None, gt_data.get("vat_amount")))
        _update_acc(metrics.field_accuracy, "total_amount_after_vat", _compare_currency(invoice.total_amount_after_vat.value if invoice.total_amount_after_vat else None, gt_data.get("total_amount_after_vat")))
        
        qr_expected = gt_data.get("qr_code_present", False)
        _update_acc(metrics.field_accuracy, "qr_code_present", invoice.qr_code_present == qr_expected)
        
        gt_items = gt_data.get("line_items", [])
        ext_items = invoice.line_items
        
        _update_acc(metrics.line_item_metrics, "row_count_match", len(ext_items) == len(gt_items))
        
        for i in range(min(len(gt_items), len(ext_items))):
            gt_item = gt_items[i]
            ext_item = ext_items[i]
            
            _update_acc(metrics.line_item_metrics, "description", _compare_text(ext_item.description.value if ext_item.description else None, gt_item.get("description")))
            _update_acc(metrics.line_item_metrics, "quantity", _compare_currency(ext_item.quantity.value if ext_item.quantity else None, gt_item.get("quantity")))
            _update_acc(metrics.line_item_metrics, "unit", _compare_text(ext_item.unit.value if ext_item.unit else None, gt_item.get("unit")))
            _update_acc(metrics.line_item_metrics, "unit_price", _compare_currency(ext_item.unit_price.value if ext_item.unit_price else None, gt_item.get("unit_price")))
            _update_acc(metrics.line_item_metrics, "vat_rate", _compare_currency(ext_item.vat_rate.value if ext_item.vat_rate else None, gt_item.get("vat_rate")))
            _update_acc(metrics.line_item_metrics, "vat_amount", _compare_currency(ext_item.vat_amount.value if ext_item.vat_amount else None, gt_item.get("vat_amount")))
            _update_acc(metrics.line_item_metrics, "amount", _compare_currency(ext_item.amount.value if ext_item.amount else None, gt_item.get("amount")))
            _update_acc(metrics.line_item_metrics, "formula_correct", "mismatch" not in str(result_state.get("validation_errors", [])))

    return metrics

def compare_benchmarks(metrics_a: BenchmarkMetrics, metrics_b: BenchmarkMetrics):
    print("\n" + "="*80)
    print("                A/B BENCHMARK COMPARISON REPORT")
    print("="*80)
    
    print(f"{'Metric':<30} | {metrics_a.name:<20} | {metrics_b.name:<20} | Diff")
    print("-" * 80)
    
    def _diff_str(val_a, val_b, is_percent=False):
        diff = val_b - val_a
        sign = "+" if diff > 0 else ""
        fmt = f"{sign}{diff:.2f}"
        if is_percent: return fmt + "%"
        return fmt
    
    a_fail = (metrics_a.validation_failure_count / metrics_a.total_invoices) * 100 if metrics_a.total_invoices else 0
    b_fail = (metrics_b.validation_failure_count / metrics_b.total_invoices) * 100 if metrics_b.total_invoices else 0
    print(f"{'Validation Failure Rate':<30} | {a_fail:>18.2f}% | {b_fail:>18.2f}% | {_diff_str(a_fail, b_fail, True)}")
    
    a_time = metrics_a.total_time_ms / metrics_a.total_invoices if metrics_a.total_invoices else 0
    b_time = metrics_b.total_time_ms / metrics_b.total_invoices if metrics_b.total_invoices else 0
    print(f"{'Average Latency (ms)':<30} | {a_time:>18.2f} | {b_time:>18.2f} | {_diff_str(a_time, b_time)} ms")
    
    a_gem_calls = metrics_a.gemini_calls / metrics_a.total_invoices if metrics_a.total_invoices else 0
    b_gem_calls = metrics_b.gemini_calls / metrics_b.total_invoices if metrics_b.total_invoices else 0
    print(f"{'Avg Gemini Calls / Invoice':<30} | {a_gem_calls:>18.2f} | {b_gem_calls:>18.2f} | {_diff_str(a_gem_calls, b_gem_calls)}")
    
    print("\n--- Field Accuracy ---")
    for k in metrics_a.field_accuracy.keys():
        v_a = metrics_a.field_accuracy[k]
        v_b = metrics_b.field_accuracy[k]
        acc_a = (v_a[0] / v_a[1] * 100) if v_a[1] > 0 else 0
        acc_b = (v_b[0] / v_b[1] * 100) if v_b[1] > 0 else 0
        print(f"{k:<30} | {acc_a:>18.2f}% | {acc_b:>18.2f}% | {_diff_str(acc_a, acc_b, True)}")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(dataset_path, exist_ok=True)
    
    print("Running Configuration A: Baseline (PP-OCRv6 only)...")
    metrics_baseline = run_evaluation(dataset_path, enable_gemini=False, name="Config A (Baseline)")
    
    print("\nRunning Configuration B: Baseline + Gemini Verification...")
    metrics_gemini = run_evaluation(dataset_path, enable_gemini=True, name="Config B (Gemini)")
    
    compare_benchmarks(metrics_baseline, metrics_gemini)
