import org.apache.spark.rdd.RDD
import scala.math.log
object TFIDF {
  // flatMap each email into ((docId, word), 1) pairs, reduceByKey to count,
  // then normalize by max count per document
  def computeTF(emails: RDD[(Long, Email)]): RDD[(Long, Map[String, Double])] = {
    val wordPairs = emails.flatMap { case (id, email) =>
      Preprocessing.cleanText(email.text).map(word => ((id, word), 1.0))
    }
    val wordCounts = wordPairs.reduceByKey(_ + _)
    wordCounts.map { case ((id, word), count) => (id, (word, count)) }
      .groupByKey()
      .map { case (id, words) =>
        val countsMap = words.toMap
        if (countsMap.isEmpty) {
          (id, Map.empty[String, Double])
        } else {
          val maxCount = countsMap.values.max
          (id, countsMap.map { case (w, c) => (w, c / maxCount) })
        }
      }
  }
  // MapReduce to compute Document Frequency
  //   Map: (word, 1) for every word
  //   Reduce: sum counts to get how many documents contain each word
  def computeDF(tf: RDD[(Long, Map[String, Double])]): Map[String, Double] = {
    tf.flatMap { case (_, wordMap) =>
        wordMap.keys.map(word => (word, 1.0))
      }
      .reduceByKey(_ + _)
      .collectAsMap()
      .toMap
  }
  // compute IDF = log(N / DF) and multiply TF * IDF
  def computeTFIDF(
    tf: RDD[(Long, Map[String, Double])],
    df: Map[String, Double],
    totalDocs: Long
  ): RDD[(Long, Map[String, Double])] = {
    val idf = df.map { case (word, docFreq) =>
      (word, log(totalDocs.toDouble / docFreq))
    }
    val idfBroadcast = tf.sparkContext.broadcast(idf)

    tf.map { case (id, tfMap) =>
      val localIdf = idfBroadcast.value
      val tfidfMap = tfMap.map { case (word, tfVal) =>
        (word, tfVal * localIdf.getOrElse(word, 0.0))
      }
      (id, tfidfMap)
    }
  }
  def compute(
    emails: RDD[(Long, Email)],
    totalDocs: Long
  ): (RDD[(Long, Map[String, Double])], Int) = {
    val tf = computeTF(emails)
    val df = computeDF(tf)
    val tfidf = computeTFIDF(tf, df, totalDocs)
    (tfidf, df.size)
  }
  def labelVectors(
    tfidfRDD: RDD[(Long, Map[String, Double])],
    emailRDD: RDD[(Long, Email)]
  ): RDD[(Long, (Map[String, Double], String))] = {
    tfidfRDD.join(emailRDD).map { case (id, (vec, email)) =>
      (id, (vec, email.label))
    }
  }
}