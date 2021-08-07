package xai.hlta

import org.latlab.io.bif.BifParser
import java.io.FileInputStream
import org.latlab.model.LTM
import org.latlab.util.DataSet
import collection.JavaConversions._
import java.io.PrintWriter
import scala.io.Source
import xai.util.Tree
import xai.util.TreeList
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.Path
import xai.util.FileHelpers
import xai.util.Arguments
import scala.collection.GenSeq
import org.apache.commons.text.StringEscapeUtils
import org.slf4j.LoggerFactory



/**
 * BuildWebsite saves all files in .js so no ajax.get is required
 * Ajax.get will require a server
 * No ajax call allows to read the website as normal file
 */
object BuildWebsite{

  val packageName = "xai/hlta"
  
  val logger = LoggerFactory.getLogger(BuildWebsite.getClass)
  
  /**
   * for external call
   */
  def apply(outputName: String, title: String, topicTree: TopicTree = null, catalog: DocumentCatalog = null, 
      docNames: Seq[String] = null, docUrls: Seq[String] = null){
    if(topicTree!=null) topicTree.saveAsJs(outputName + ".nodes.js", jsVarName = "nodes")
    if(catalog!=null) catalog.saveAsJs(outputName + ".topics.js", jsVarName = "topicMap")
    if(docNames!=null) writeDocNames(docNames, s"${outputName}.titles.js", docUrls = docUrls)
    writeHtmlOutput(title, outputName, outputName + ".html")
    val dir = Paths.get(outputName).getParent
    copyAssetFiles(dir)
  }
  
  /**
   * Write document labels
   */
  def writeDocNames(docNames: GenSeq[String], outputFile: String, docUrls: Seq[String] = null) = {
    implicit class Escape(str: String){
      def escape = str.replaceAll("\\\\", "\\\\\\\\").replaceAll("\"", "\\\\\"").replaceAll("\n", " ").replaceAll("\r", " ")
    }
    
    val writer = new PrintWriter(outputFile)
    writer.println("var documents = [")
    if(docUrls==null)    writer.println(docNames.map{docName=>"\""+docName.escape+"\""}.mkString(",\n"))
    else   writer.println(docNames.zip(docUrls).map{case(docName, docUrl) => 
        if(docUrl.isEmpty) "\""+docName.escape+"\""
        else "[\""+docName.escape+"\",\""+docUrl.escape+"\"]"
      }.mkString(",\n"))
    writer.println("]")
    writer.close
  }
  
//  /**
//   * Translate csv to js
//   */
//  def translateCsv2Js(inputFile: String, outputFile: String, encoding: String = "UTF-8") = {
//    implicit class Escape(str: String){
//      def escape = str.replaceAll("\\\\", "\\\\\\\\").replaceAll("\"", "\\\\\"").replaceAll("\n", " ").replaceAll("\r", " ")
//    }
//    import com.github.tototoshi.csv._
//    val reader = CSVReader.open(inputFile, encoding)
//    val (List(header), data) = reader.all().splitAt(1)
//
//    val writer = new PrintWriter(outputFile, encoding)
//    writer.println("var fieldnames = [")
//    writer.println("\""+header.map(_.escape).mkString("\", \"")+"\"")
//    writer.println("];")
//    writer.println("var documents = [")
//    writer.println(data.map{d=>"[\""+d.map(_.escape).mkString("\", \"")+"\"]"}.mkString(",\n"))
//    writer.println("];")
//    writer.close
//  }
  
  /**
   * Write main webpage
   */
  def writeHtmlOutput(title: String, outputName: String, outputFile: String) = {
    
    implicit class Escape(str: String){
      def escape = StringEscapeUtils.escapeHtml4(str)
    }
    
    val template = Source.fromInputStream(
      this.getClass.getResourceAsStream(s"/$packageName/template.html"))
      .getLines.mkString("\n")

    val writer = new PrintWriter(outputFile)

    def replace(s: String, target: String) = {
      s.replaceAll(s""""${target}"""", s""""${outputName}.${target}"""")
    }

    var content = Seq("nodes.js", "topics.js", "titles.js")
      .foldLeft(template)(replace)
    content = content.replaceAll("<!-- title-placeholder -->", title.escape)

    writer.print(content)
    writer.close
  }

  /**
   * Copy image and JavaScript files
   */
  def copyAssetFiles(basePath: Path) = {
    val baseDir = Option(basePath).getOrElse(Paths.get("."))
    val assetDir = baseDir.resolve("lib")
    val fontsDir = baseDir.resolve("fonts")

    if (!Files.exists(assetDir))
      Files.createDirectories(assetDir)

    if (!Files.exists(fontsDir))
      Files.createDirectories(fontsDir)

    def copyTo(source: String, dir: Path, target: String) = {
      val input = this.getClass.getResourceAsStream(source)
      logger.debug(s"Copying from resource ${source} to file ${dir.resolve(target)}")
      Files.copy(input, dir.resolve(target), StandardCopyOption.REPLACE_EXISTING)
    }

    def copy(obj: Object, dir: Path) = obj match {
      case (source: String, target: String) => copyTo(source, dir, target)
      case (source: String) => {
        val index = source.lastIndexOf("/")
        val name = if (index < 0) source else source.substring(index + 1)
        copyTo(source, dir, name)
      }
      //println(s"Copy from resource, totally done")
    }

    Seq(
      (s"/$packageName/jquery-2.2.3.min.js", "jquery.min.js"),
      s"/$packageName/jstree.min.js",
      //      s"/$pack/jquery.magnific-popup.min.js",
      s"/$packageName/jquery.tablesorter.min.js",
      s"/$packageName/jquery.tablesorter.widgets.js",
      s"/$packageName/magnific-popup.css",
      s"/$packageName/custom.js",
      s"/$packageName/custom.css",
      //      s"/$pack/tablesorter/blue/asc.gif",
      //      s"/$pack/tablesorter/blue/bg.gif",
      //      s"/$pack/tablesorter/blue/desc.gif",
      //      (s"/$pack/tablesorter/blue/style.css", "tablesorter.css"),
      (s"/$packageName/tablesorter/themes/theme.bootstrap.css", "tablesorter.css"),
      //      s"/$pack/tablesorter/themes/bootstrap-black-unsorted.png",
      //      s"/$pack/tablesorter/themes/bootstrap-white-unsorted.png",
      s"/$packageName/jstree/themes/default/style.min.css",
      s"/$packageName/jstree/themes/default/32px.png",
      s"/$packageName/jstree/themes/default/40px.png",
      s"/$packageName/jstree/themes/default/throbber.gif",
      s"/$packageName/ie10-viewport-bug-workaround.css",
      s"/$packageName/ie10-viewport-bug-workaround.js",
      s"/$packageName/bootstrap.min.css",
      s"/$packageName/bootstrap.min.js")
      .foreach(p => copy(p, assetDir))

    Seq(s"/$packageName/bootstrap/fonts/glyphicons-halflings-regular.eot",
      s"/$packageName/bootstrap/fonts/glyphicons-halflings-regular.svg",
      s"/$packageName/bootstrap/fonts/glyphicons-halflings-regular.ttf",
      s"/$packageName/bootstrap/fonts/glyphicons-halflings-regular.woff",
      s"/$packageName/bootstrap/fonts/glyphicons-halflings-regular.woff2")
      .foreach(p => copy(p, fontsDir))
  }
}


/**
 * Jstree is a javascript library, see www.jstree.com
 */
object JstreeWriter{
  
  val logger = LoggerFactory.getLogger(JstreeWriter.getClass)
  
  /**
   * Describes how topic is presented in the jstree
   */
  case class Node(id: String, label: String, data: Map[String, Any])
  
  def writeJs[A](roots: Seq[Tree[A]], outputFile: String, jsVarName: String, jstreeContent: A => Node){
    
    implicit class Escape(str: String){
      def escape = str.replaceAll("\\\\", "\\\\\\\\").replaceAll("\"", "\\\\\"").replaceAll("\n", " ").replaceAll("\r", " ")
    }

    def _treeToJs(tree: Tree[A], indent: Int): String = {
      val node = jstreeContent(tree.value)
      val children = tree.children.map(_treeToJs(_, indent + 4))
      val js = """{
      |  id: "%s", text: "%s", data: { %s }, children: [%s]
      |}""".format(node.id.escape, node.label.escape, node.data.map{
        case (variable: String, value: String) => variable.escape+": \""+value.escape+"\""
        case (variable: String, value: Number) => variable.escape+": "+value
        case (variable: String, value) => variable.escape+": "+value
        }.mkString(", "), children.mkString(", "))
        .replaceAll(" +\\|", " " * indent)
      js
    }
    
    val writer = new PrintWriter(outputFile)
    writer.print("var "+jsVarName+" = [")
    writer.println(roots.map(_treeToJs(_, 0)).mkString(", "))
    writer.println("];")
    writer.close
  }

  def writeJson[A](roots: Seq[Tree[A]], outputFile: String, jstreeContent: A => Node){
    
    implicit class Escape(str: String){
      def escape = str.replaceAll("\\\\", "\\\\\\\\").replaceAll("\"", "\\\\\"").replaceAll("\n", " ").replaceAll("\r", " ")
    }
    
    def _treeToJson(tree: Tree[A], indent: Int): String = {
      val node = jstreeContent(tree.value)
      val children = tree.children.map(_treeToJson(_, indent + 4))
      val json = """{
      |  "id": "%s", "text": "%s", "data": {%s}, "children": [%s]
      |}""".format(node.id, node.label, node.data.map{
        case (variable: String, value: Number) => "\""+variable.escape+"\": "+value
        case (variable: String, value: String) => "\""+variable.escape+"\": \""+value+"\""
        case (variable: String, value) => "\""+variable.escape+"\": \""+value+"\""
        }.mkString(", "),  children.mkString(", "))
        .replaceAll(" +\\|", " " * indent)
      json
    }
    
    logger.debug(s"will write json file, open writeJson: " + outputFile)
    val writer = new PrintWriter(outputFile)
    writer.print("[")
    writer.println(roots.map(_treeToJson(_, 0)).mkString(", "))
    writer.println("]")
    logger.debug(s"writeJson done")
    writer.close
  }

  /**
   * Plain html, no jstree plugin
   * TODO: change this to be the format of Peixian's html
   */
  def writeSimpleHtml[A](roots: Seq[Tree[A]], outputFile: String, jstreeContent: A => Node){
    
    implicit class Escape(str: String){
      def escape = StringEscapeUtils.escapeHtml4(str)
    }

    def _treeToHtml(tree: Tree[A], indent: Int): String = {
      val node = jstreeContent(tree.value)
      val start = """<li class="jstree-open" id="%s" >""".format(node.id.escape)
      val content = node.label.escape
      val end = "</li>"
  
      if (tree.children.isEmpty)
        " " * indent + start + content + end + "\n"
      else {
        val childIndent = indent + 2
  
        " " * indent + start + content + "\n" +
          " " * childIndent + "<ul>" + "\n" +
          tree.children.map(_treeToHtml(_, childIndent)).reduce(_ + _) +
          " " * childIndent + "</ul>" + "\n" +
          " " * indent + end + "\n"
      }
    }
    
    val writer = new PrintWriter(outputFile)
    writer.println(roots.map(_treeToHtml(_, 0)).reduce(_ + _))
    writer.close
  }
}
