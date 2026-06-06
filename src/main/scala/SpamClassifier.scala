import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.PrintWriter
object SpamClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("SpamEmailClassifier")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val dataPath = if (args.length > 0) args(0) else "data/spam_Emails_data.csv"
    val k = if (args.length > 1) args(1).toInt else 5
    val sampleSize = if (args.length > 2) args(2).toInt else 0
    val outputPath = if (args.length > 3) new Path(args(3)) else new Path("output.txt")
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(outputPath)) {
      sc.stop()
      return
    }
    val startTime = System.currentTimeMillis()
    val (trainRDD, testRDD, totalEmails, spamCount, hamCount) =
      Preprocessing.loadAndSplit(sc, dataPath, sampleSize)
    val trainBoW = BagOfWords.labelVectors(BagOfWords.compute(trainRDD), trainRDD)
    val testBoW = BagOfWords.labelVectors(BagOfWords.compute(testRDD), testRDD)
    trainBoW.persist()
    val allEmails = trainRDD.union(testRDD)
    val totalDocs = allEmails.count()
    val (allTFIDF, vocabSize) = TFIDF.compute(allEmails, totalDocs)
    val trainTFIDF = TFIDF.labelVectors(allTFIDF, trainRDD)
    val testTFIDF = TFIDF.labelVectors(allTFIDF, testRDD)
    trainTFIDF.persist()
    val bowPredictions = KNN.classify(trainBoW, testBoW, k)
    val tfidfPredictions = KNN.classify(trainTFIDF, testTFIDF, k)
    val totalTime = System.currentTimeMillis() - startTime
    val out = fs.create(outputPath)
    val writer = new PrintWriter(out)
    writer.println(s"$totalEmails emails ($spamCount spam, $hamCount ham), k=$k, vocab=$vocabSize")
    Evaluation.evaluate("BoW", bowPredictions, writer)
    Evaluation.evaluate("TF-IDF", tfidfPredictions, writer)
    writer.println(s"Time: ${totalTime / 1000}s")
    writer.close()
    sc.stop()
  }
}
