import org.apache.spark.rdd.RDD
import scala.math.sqrt
object KNN {
  def cosineSimilarity(v1: Map[String, Double], v2: Map[String, Double]): Double = {
    val commonKeys = v1.keySet.intersect(v2.keySet)
    if (commonKeys.isEmpty) return 0.0
    val dotProduct = commonKeys.map(k => v1(k) * v2(k)).sum
    val mag1 = sqrt(v1.values.map(v => v * v).sum)
    val mag2 = sqrt(v2.values.map(v => v * v).sum)
    if (mag1 == 0.0 || mag2 == 0.0) 0.0
    else dotProduct / (mag1 * mag2)
  }
  // Uses mapPartitions so that the train array is loaded once per partition then reused for every test email in that partition
  def classify(
    trainVectors: RDD[(Long, (Map[String, Double], String))],
    testVectors: RDD[(Long, (Map[String, Double], String))],
    k: Int
  ): Array[(Long, String, String)] = {
    val trainCollected = trainVectors.collect()
    val trainBroadcast = trainVectors.sparkContext.broadcast(trainCollected)

    val predictions = testVectors.mapPartitions { partition =>
      val localTrain = trainBroadcast.value
      partition.map { case (testId, (testVec, actualLabel)) =>
        val similarities = localTrain.map { case (_, (trainVec, trainLabel)) =>
          (cosineSimilarity(testVec, trainVec), trainLabel)
        }
        val topK = similarities.sortBy(-_._1).take(k)
        val votes = topK.groupBy(_._2).map { case (label, arr) => (label, arr.length) }
        val predictedLabel = votes.maxBy(_._2)._1

        (testId, actualLabel, predictedLabel)
      }
    }
    predictions.collect()
  }
}
