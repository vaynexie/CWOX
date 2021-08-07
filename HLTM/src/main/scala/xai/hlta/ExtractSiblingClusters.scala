package xai.hlta

import org.latlab.model.{BayesNet, LTM}

import java.io.PrintWriter
import java.nio.file._
import xai.util.manage
import xai.util.Reader

import scala.collection.JavaConverters._

object ExtractSiblingClusters {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("ExtractSiblingClusters model_file")
      println("ExtractSiblingClusters directory")
    } else {
      val path = Paths.get(args(0))
      if (Files.isDirectory(path))
        processDirectory(path, extractFile)
      else
        processFile(path, extractFile)
    }
  }

  def processDirectory(directory: Path, op: (String, LTM) => Unit): Unit = {
    val files = Files.list(directory).iterator().asScala.filter(_.toString.endsWith(".bif"))
    files.toList.par.foreach(processFile(_, op))
  }

  def processFile(file: Path, op: (String, LTM) => Unit) = {
    val filename = file.toString
    try {
      val model = Reader.readLTM(filename)

      op(filename, model)
    } catch {
      case e: Exception =>
        println(s"Cannot process ${file}: ${e.getMessage}")
    }
  }


  def extractClusterMap(model: LTM) = {
    val observedNodes = model.getManifestVars.asScala.toList.map(model.getNode)
    val pairs = observedNodes.map(n => (n.getParent.getName, n.getName))
    pairs.groupBy(_._1)
  }

  def showClusters(clusterMap: Map[String, List[(String, String)]]) =
    clusterMap.map(g => s"${g._1}: ${g._2.map(_._2).mkString(", ")}").mkString("\n")

  def extractFile(filename: String, model: LTM) = {
    val output = filename.replace(".bif", "-siblings.txt")
    manage(new PrintWriter(output)) { writer =>

      writer.println(showClusters(extractClusterMap(model)))
      println(s"${output} saved.")
    }
  }
}

object InspectClusters {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("InspectClusters model_file")
      println("InspectClusters directory")
    } else {
      val path = Paths.get(args(0))
      if (Files.isDirectory(path))
        ExtractSiblingClusters.processDirectory(path, inspect)
      else
        ExtractSiblingClusters.processFile(path, inspect)
    }
  }

  def inspect(filename: String, model: LTM) = {
    val groups = ExtractSiblingClusters.extractClusterMap(model).map(_._2.map(_._2))
    val sizes = groups.map(_.size)
    val average = sizes.sum.toDouble / sizes.size
    println(s"${filename}: ${average}")
  }
}
