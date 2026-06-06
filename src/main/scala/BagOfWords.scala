import org.apache.spark.rdd.RDD
object BagOfWords {
  // flatMap each email into (docId, word) pairs, then reduceByKey to count
  def compute(emails: RDD[(Long, Email)]): RDD[(Long, Map[String, Double])] = {
    val wordPairs = emails.flatMap { case (id, email) =>
      Preprocessing.cleanText(email.text).map(word => ((id, word), 1.0))
    }
    val wordCounts = wordPairs.reduceByKey(_ + _)
    wordCounts.map { case ((id, word), count) => (id, (word, count)) }
      .groupByKey()
      .map { case (id, words) => (id, words.toMap) }
  }
  def labelVectors(
    bowRDD: RDD[(Long, Map[String, Double])],
    emailRDD: RDD[(Long, Email)]
  ): RDD[(Long, (Map[String, Double], String))] = {
    bowRDD.join(emailRDD).map { case (id, (bow, email)) =>
      (id, (bow, email.label))
    }
  }
}