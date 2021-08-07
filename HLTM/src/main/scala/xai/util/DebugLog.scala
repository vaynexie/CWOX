package xai.util

import scala.collection.JavaConverters._
import org.latlab.model.LTM
import org.latlab.util.{DataSet, Variable}
import org.latlab.util.ScoreCalculator.computeLoglikelihood
import xai.hlta.ExtractSiblingClusters.{extractClusterMap, showClusters}

import java.util

object DebugLog {
  // def printSeparator() = println("*************************************************")
  def printSeparator() = {}

  def logUDTest(bestPair: util.ArrayList[Variable], newPair: util.ArrayList[Variable],
                m1: LTM, m2: LTM, data: DataSet) = {
    val weight = data.getTotalWeight

    def compute(model: LTM) = {
      val d = model.computeDimension()
      val ll = computeLoglikelihood(model, data)
      val logN = Math.log(weight)
      val bic = ll - d * logN / 2.0

      (bic, ll, d, logN)
    }

    def computeAndDisplay(label: String, model: LTM) = {
      val (bic, ll, d, logN) = compute(model)

      println(label)
      println(showClusters(extractClusterMap(model)))
      println(s"BIC: ${bic}, LL: ${ll}, dim: ${d}, lnN: ${logN}")

      (bic, ll, d, logN)
    }

    val (bic1, _, _, _) = computeAndDisplay("model 1", m1)
    val (bic2, _, _, _) = computeAndDisplay("model 2", m2)

    println(s"BIC2 - BIC1: ${bic2 - bic1}")
    println("Best pair: " + bestPair.asScala.map(_.getName).mkString(", "))
  }

  def logNewCluster(reason: String, subModel: LTM) = {
    println(s"[${reason}] New cluster: ${
      subModel.getManifestVars.asScala.map(_.getName).mkString(", ")}")
    printSeparator()
  }
}
