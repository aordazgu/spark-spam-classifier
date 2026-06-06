import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
case class Email(id: Long, label: String, text: String)
object Preprocessing {
  val stopwords: Set[String] = Set(
    "img", "src", "http", "https", "www", "com", "href", "nbsp",
    "quot", "gt", "lt", "amp", "html", "div", "span", "class",
    "style", "width", "height", "border", "font", "color", "size",
    "table", "bgcolor", "cellpadding", "cellspacing", "align",
    "valign", "mailto", "url", "org", "net", "edu"
  )
  def cleanText(text: String): Array[String] = {
    text.toLowerCase
      .replaceAll("[^a-z\\s]", " ")
      .split("\\s+")
      .filter(w => w.nonEmpty && !stopwords.contains(w))
  }
  def parseCSV(lines: Array[String]): Array[(String, String)] = {
    val results = scala.collection.mutable.ArrayBuffer[(String, String)]()
    var i = 1
    while (i < lines.length) {
      val line = lines(i)
      if (line.startsWith("Spam,") || line.startsWith("Ham,")) {
        val commaIdx = line.indexOf(',')
        val label = line.substring(0, commaIdx)
        val rest = line.substring(commaIdx + 1)
        if (rest.startsWith("\"")) {
          val textBuilder = new StringBuilder(rest.substring(1))
          i += 1
          var closed = rest.endsWith("\"") && !rest.endsWith("\\\"")
          while (i < lines.length && !closed) {
            val nextLine = lines(i)
            if (nextLine.endsWith("\"")) {
              textBuilder.append(" ").append(nextLine.dropRight(1))
              closed = true
            } else {
              textBuilder.append(" ").append(nextLine)
            }
            i += 1
          }
          results += ((label, textBuilder.toString))
        } else {
          results += ((label, rest))
          i += 1
        }
      } else {
        i += 1
      }
    }
    results.toArray
  }
  def loadAndSplit(
    sc: SparkContext,
    dataPath: String,
    sampleSize: Int = 0
  ): (RDD[(Long, Email)], RDD[(Long, Email)], Int, Int, Int) = {
    val rawLines = sc.textFile(dataPath).collect()
    val parsed = parseCSV(rawLines)
    val emailsArray = parsed.zipWithIndex.map { case ((label, text), idx) =>
      (idx.toLong, Email(idx.toLong, label, text))
    }
    val data = if (sampleSize > 0) {
      scala.util.Random.shuffle(emailsArray.toSeq).take(sampleSize).toArray
    } else {
      emailsArray
    }
    val totalEmails = data.length
    val spamCount = data.count(_._2.label == "Spam")
    val hamCount = data.count(_._2.label == "Ham")
    val shuffled = scala.util.Random.shuffle(data.toSeq).toArray
    val splitIdx = (totalEmails * 0.8).toInt
    val trainArray = shuffled.take(splitIdx)
    val testArray = shuffled.drop(splitIdx)
    val trainRDD = sc.parallelize(trainArray)
    val testRDD = sc.parallelize(testArray)
    trainRDD.persist()
    testRDD.persist()
    (trainRDD, testRDD, totalEmails, spamCount, hamCount)
  }
}