"""ONNX export ve doğrulama aracı (PyTorch vs ONNX Runtime)."""
import os
import argparse
import numpy as np
import torch
import onnxruntime as ort

from network import Connect4Net
from network_v2 import Connect4NetV2
from train import load_checkpoint


def export_to_onnx(model, output_path="model.onnx", dynamic_batch=True):
    """PyTorch modelini ONNX formatına aktarır."""
    model.eval()
    dummy_input = torch.randn(1, 3, 6, 7, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Model başarıyla ONNX formatına aktarıldı: {output_path}")


def verify_onnx(model, onnx_path="model.onnx", tolerance=1e-5):
    """PyTorch ve ONNX Runtime çıktılarının aynı girdi için birebir eşleştiğini doğrular."""
    model.eval()

    # ONNX Runtime oturumu oluştur (CPU optimizasyonlu)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    # Test girdisi
    test_input = np.random.randn(2, 3, 6, 7).astype(np.float32)

    # 1. PyTorch Çıktısı
    with torch.no_grad():
        pt_policy, pt_value = model(torch.from_numpy(test_input))
        pt_policy = pt_policy.numpy()
        pt_value = pt_value.numpy()

    # 2. ONNX Runtime Çıktısı
    ort_inputs = {"input": test_input}
    ort_policy, ort_value = session.run(None, ort_inputs)

    # Karşılaştırma
    policy_match = np.allclose(pt_policy, ort_policy, atol=tolerance)
    value_match = np.allclose(pt_value, ort_value, atol=tolerance)

    max_p_diff = np.max(np.abs(pt_policy - ort_policy))
    max_v_diff = np.max(np.abs(pt_value - ort_value))

    print(f"Doğrulama Sonuçları:")
    print(f"  Policy Eşleşmesi: {'BAŞARILI' if policy_match else 'BAŞARISIZ'} (Maks fark: {max_p_diff:.2e})")
    print(f"  Value Eşleşmesi:  {'BAŞARILI' if value_match else 'BAŞARISIZ'} (Maks fark: {max_v_diff:.2e})")

    assert policy_match and value_match, "ONNX ve PyTorch çıktıları eşleşmedi!"
    print("Her iki motorun çıktıları kusursuz eşleşti.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Yüklenecek checkpoint (.pt)")
    parser.add_argument("--v2", action="store_true", help="Connect4NetV2 (116K) modelini kullan")
    parser.add_argument("--output", type=str, default="test_model.onnx", help="Çıktı ONNX yolu")
    args = parser.parse_args()

    # Model tipi seçimi
    if args.v2:
        model = Connect4NetV2()
        print("Model Mimarisi: Connect4NetV2 (116K)")
    else:
        model = Connect4Net()
        print("Model Mimarisi: Connect4Net (465K)")

    if args.checkpoint and os.path.exists(args.checkpoint):
        it = load_checkpoint(model, args.checkpoint)
        print(f"Checkpoint yüklendi: {args.checkpoint} (İter: {it})")

    export_to_onnx(model, output_path=args.output)
    verify_onnx(model, onnx_path=args.output)

    # Test sonrası geçici onnx dosyasını temizle (istenirse saklanabilir)
    if os.path.exists(args.output):
        os.remove(args.output)
        print("Test tamamlandı, geçici ONNX dosyası temizlendi.")
