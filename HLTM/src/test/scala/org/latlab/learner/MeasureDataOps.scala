package org.latlab.learner

import xai.util.Reader
import xai.util.Timer.time

import scala.util.Random

import scala.collection.JavaConverters._

object MeasureDataOps {

  def main(args: Array[String]): Unit = {
    val filename = args(0)
    val sparseData = time("Reading data")(Reader.readData(filename).toTupleSparseDataSet())

    checkSampling(sparseData, args.drop(1))
//    checkSparseToDense(sparseData)
//    checkProjection(sparseData, args.drop(1))
  }

  def checkSparseToDense(sparseData: SparseDataSet) = {
    def original(sparseData: SparseDataSet) = {
      val wholeData = time("Converting to whole dense data")(sparseData.getWholeDenseData())
      println(wholeData.getTotalWeight)
    }

    def updated(sparseData: SparseDataSet) = {
      val converted = time("Converting with DataOps")(DataOps.convertToDataSet(sparseData))
      println(converted.getTotalWeight)
    }

    for (i <- 0 until 5) {
      updated(sparseData)
      original(sparseData)
    }
  }

  def checkProjection(sparseData: SparseDataSet, args: Array[String]) = {

    val size = if (args.length == 0) 10 else args(0).toInt
    println("Subset size: " + size)

    val dataSet = time("Converting to Dense data set with DataOps")(DataOps.convertToDataSet(sparseData))
    val subset = new java.util.ArrayList(Random.shuffle(dataSet.getVariables.toList).take(size).asJava)

    def original(sparseData: SparseDataSet) = {
      val projected = time("Projecting with method of DataSet")(dataSet.project(subset))
      println(s"weight: ${projected.getTotalWeight} | size: ${projected.getData.size()}")
    }

    def updated(sparseData: SparseDataSet) = {
      val projected = time("Projecting with DataOps")(DataOps.project(dataSet, subset))
      println(s"weight: ${projected.getTotalWeight} | size: ${projected.getData.size()}")
    }

    for (i <- 0 until 50) {
      updated(sparseData)
      original(sparseData)
    }

  }

  def checkSampling(sparseData: SparseDataSet, args: Array[String]) = {

    val size = if (args.length == 0) 1000 else args(0).toInt
    println("Sample size: " + size)

    val dataSet = time("Converting to Dense data set with DataOps")(DataOps.convertToDataSet(sparseData))

    def original() = {
      val sampled = time("Sampling with method of DataSet")(
        dataSet.sampleWithReplacement(size))
      println(s"weight: ${sampled.getTotalWeight} | entries: ${sampled.getData.size()}")
    }

    def updated() = {
      val sampled = time("Sampling with DataOps")(
        DataOps.sampleWithReplacement(dataSet, size))
      println(s"weight: ${sampled.getTotalWeight} | entries: ${sampled.getData.size()}")
    }

    for (i <- 0 until 5) {
      updated()
      original()
    }
  }
}
