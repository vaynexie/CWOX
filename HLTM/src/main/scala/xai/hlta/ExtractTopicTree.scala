package xai.hlta

import java.io.PrintWriter

import xai.util.{Arguments, Data, FileHelpers, Reader, Tree, manage}
import java.nio.file.{Files, Path, Paths}

import scala.io.Source
import org.latlab.model.LTM
import xai.hlta.HLTA._
import org.latlab.util.DataSet
import org.slf4j.LoggerFactory
import xai.util

import scala.util.matching.Regex.Match
import scala.collection.JavaConverters._

object ExtractTopicTree {
  class Conf(args: Seq[String]) extends Arguments(args) {
    banner("Usage: tm.hlta.ExtractTopicTree [OPTION]... name model_file")
    val name = trailArg[String](descr = "Name of the topic tree file to be generated")
    val model = trailArg[String](descr = "Model file (e.g. model.bif)")

    val title = opt[String](default = Some("Topic Tree"), descr = "Title in the topic tree")
    val layer = opt[List[Int]](descr = "Layer number, i.e. --layer 1 3")
    val keywords = opt[Int](default = Some(100), descr = "number of keywords for each topic (default: 100)")
//    val keywordsProb = opt[Boolean](default = Some(false), descr = "show probability of individual keyword")
    val tempDir = opt[String](default = Some("topic_output"),
      descr = "Temporary output directory for extracted topic files (default: topic_output)")

    verify
    checkDefaultOpts()
  }

  val logger = LoggerFactory.getLogger(ExtractTopicTree.getClass)

  def main(args: Array[String]) {
    val conf = new Conf(args)

    val topicTree =  {
      //Broad defined topic do not recompute parameters
      //Thus, no data are required
      val model = Reader.readModel(conf.model())
      logger.debug(s"will broad")
      broad(model, conf.layer.toOption, conf.keywords(), false)
    }
    logger.info("Topic tree extraction is done.")
    logger.debug(s"narrow or broad done. will BuildWebsite")

    //BuildWebsite generates file in .js format
    BuildWebsite(conf.name(), conf.title(), topicTree)
    //Additionally generates .json file
    topicTree.saveAsJson(conf.name()+".nodes.json")
    topicTree.saveAsSimpleHtml(conf.name()+".simple.html")
    logger.info("The topic tree is available at "+conf.name()+".html")
    logger.debug(s"saveAsJson done. filename " + conf.name() + ".nodes.json")

    RemoveProbabilities.removeFromFile(Path.of(s"${conf.name()}.nodes.js"))
  }

  def broad(model: LTM, layer: Option[List[Int]] = None, keywords: Int = 7, keywordsProb: Boolean = false) = {
//    val output = Paths.get(tempDir)
//    FileHelpers.mkdir(output)

    val extractor = new BroadTopicsExtractor(model, keywords, layer, keywordsProb)
    extractor.extractTopics()

  }


  private class BroadTopicsExtractor(model: LTM, keywords: Int,
      layers: Option[List[Int]] = None, outProbNum: Boolean = false, assignProb: Boolean = true){
    import org.latlab.util.Variable
    import org.latlab.reasoner.CliqueTreePropagation
    import java.util.ArrayList
    import collection.JavaConverters._
    import xai.hlta.HLTA
    import xai.util.Tree

    val _posteriorCtp = new CliqueTreePropagation(model);

    def extractTopics(): TopicTree = {
      val _varDiffLevels = model.getLevelVariables()
      val _layers = if(layers.isDefined) layers.get.sorted else (1 until _varDiffLevels.size).toList //in ascending order
      val topicNodeBank = scala.collection.mutable.Map[String, Tree[Topic]]()
      _layers.foreach { VarLevel =>
        _varDiffLevels.apply(VarLevel).map {latent =>
          val topic = topicForSingleVariable(latent)
          val descendentLatentVars = model.latentDescendentOf(latent.getName)
          val childs = descendentLatentVars.flatMap { v =>
            //remove and pops the topic from the bank
            topicNodeBank.remove(v.getName)
          }
          topicNodeBank.put(latent.getName, Tree.node[Topic](topic, childs))
        }
      }
      val topicTree = TopicTree(topicNodeBank.values.toSeq)
      topicTree.reassignLevel()
      topicTree
    }

    /**
  	 * Rewritten from printTopicsForSingleVariable
  	 */
  	def topicForSingleVariable(latent: Variable) = {
  		_posteriorCtp.clearEvidence();
  		_posteriorCtp.propagate();
  		val p = _posteriorCtp.computeBelief(latent);

  		val setNode = model.observedDescendentOf(latent.getName)
  		val globallist = SortChildren(latent, setNode, _posteriorCtp);

      val observedVarOrder = globallist.take(keywords).map{ case(v, mi) => v }

  		_posteriorCtp.clearEvidence();
  		_posteriorCtp.propagate();

			val latentArray = Array(latent);
			val card = 1; //Only consider z=1 state
			val states = Array(card);

			// set evidence for latent state
			_posteriorCtp.setEvidence(latentArray, states);
			_posteriorCtp.propagate();

			// compute posterior for each manifest variable
			val words = observedVarOrder.map{ manifest =>
			  if(outProbNum){
				  val posterior = _posteriorCtp.computeBelief(manifest);
				  val prob = if(manifest.getCardinality()>1) posterior.getCells()(1) else 0.0
				  Word(manifest.getName, prob)
			  }else
				  Word(manifest.getName)
			}

			// set evidence for latent state
			_posteriorCtp.setEvidence(latentArray, Array(0));
			_posteriorCtp.propagate();

			// compute posterior for each manifest variable
			val stateZeroWordsProbLookup = observedVarOrder.map{ manifest =>
				  val posterior = _posteriorCtp.computeBelief(manifest);
				  val prob = if(manifest.getCardinality()>1) (Math.rint(posterior.getCells()(1) * 100) / 100) else 0.0
				  (manifest.getName, prob)
			}.toMap

			val newWords = words.map{w=> new Word(w.w+" "+stateZeroWordsProbLookup(w.w), w.probability)}

      val size = p.getCells()(card);
			new Topic(name = latent.getName, words = newWords, level = None, size = Some(size), mi = None)
  	}

  	def SortChildren(latent: Variable, varSet: Seq[Variable], ctp: CliqueTreePropagation) = {
      varSet.map{ child =>
        val mi = computeMI(latent, child, ctp);
        (child, mi)
      }.sortBy(-_._2)
    }

    def computeMI(x: Variable, y: Variable, ctp: CliqueTreePropagation) = {
      val xyNodes = new java.util.ArrayList[Variable]();
      xyNodes.add(x);
      xyNodes.add(y);
      org.latlab.util.Utils.computeMutualInformation(ctp.computeBelief(xyNodes));
    }
  }


}

object RemoveProbabilities {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("RemoveProbabilities file [...] (e.g. {name}.tree.nodes.js)")
      println("RemoveProbabilities directory")
    }  else {
      val path = Paths.get(args(0))
      if (Files.isDirectory(path))
        removeFromFilesInDir(path)
      else {
        args.par.map(p => Paths.get(p)).foreach(removeFromFile)
      }

    }
  }

  def removeFromFilesInDir(directory: Path): Unit = {
    val files = Files.list(directory).iterator().asScala.filter(f => f.toString.endsWith(".nodes.js"))
    files.toList.par.foreach(removeFromFile)
  }

  def removeFromFile(file: Path) = {
    val filename = file.toString

    val content = manage(Source.fromFile(file.toFile)){ s => s.getLines().mkString("\n")}
    manage(new PrintWriter(file.toString)) { _.write(removeFromContent(content)) }
    println(s"${filename} saved.")
  }

  val rules = Seq(
    ("""(id: "(Z[2-9][0-9]+)", text: ")[^"]+"""".r, (m: Match) => s"""${m.group(1)}${m.group(2)}""""),
    ("""(id: "Z[0-9]+", text: ")[0-9]\.[0-9]{3} """.r, (m: Match) => m.group(1)),
    ("""([^:]) [0-9]\.[0-9]+""".r, (m: Match) => m.group(1))
  )

  def removeFromContent(content: String) =
    RemoveProbabilities.rules.foldLeft(content)((z, r) => r._1.replaceAllIn(z, r._2))

}