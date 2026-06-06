import java.io.PrintWriter
object Evaluation {
  def evaluate(label: String, predictions: Array[(Long, String, String)], writer: PrintWriter): Unit = {
    val total = predictions.length.toDouble
    val correct = predictions.count { case (_, actual, predicted) => actual == predicted }
    val accuracy = correct / total
    val tp = predictions.count { case (_, a, p) => a == "Spam" && p == "Spam" }.toDouble
    val tn = predictions.count { case (_, a, p) => a == "Ham" && p == "Ham" }.toDouble
    val fp = predictions.count { case (_, a, p) => a == "Ham" && p == "Spam" }.toDouble
    val fn = predictions.count { case (_, a, p) => a == "Spam" && p == "Ham" }.toDouble
    val precision = if (tp + fp > 0) tp / (tp + fp) else 0.0
    val recall = if (tp + fn > 0) tp / (tp + fn) else 0.0
    val f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0
    writer.println(f"$label: accuracy=$accuracy%.4f, precision=$precision%.4f, recall=$recall%.4f, f1=$f1%.4f")
    writer.println(f"  Confusion: TP=${tp}%.0f FP=${fp}%.0f TN=${tn}%.0f FN=${fn}%.0f")
  }
}
