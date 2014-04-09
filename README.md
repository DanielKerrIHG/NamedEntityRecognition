NamedEntityRecognition
======================

Repo for our AI project

It looks like semi/un supervised NER is relaitvely new, and most implementations use a fully supervised method of training with lots of annotated data. This may be to our bennefit. 

//Basic info on NER
http://en.wikipedia.org/wiki/Named-entity_recognition

//General info on Unsupervised Machine Learning:
http://mlg.eng.cam.ac.uk/zoubin/papers/ul.pdf

# **Start here, this describes the steps for NER. Background information can be found in chapters 3 and 5. ALSO
# there is source code here: http://www.nltk.org/
http://www.nltk.org/book3/ch07.html

// **Here you can download the source files for the Stanford NER (implemented in java):
http://nlp.stanford.edu/software/CRF-NER.shtml

// Lingpipe is an NLP implementation with NER module. It has a book associated with it (much like the nltk)
http://alias-i.com/lingpipe-book/lingpipe-book-0.5.pdf
  //Ch. 6 is about character lang. models, Ch. 7 is about tokenized language models, **ch 9 is about classifiers has 
  // useful information about scoring classifier performance, Ch 10 is bayes classifier, ch 11 is about tagging...

//Short powerpoint going over the steps of an semi-supervised algorithm for NER:
http://www.cs.columbia.edu/~jebara/6772/proj/an2505_project.pdf

//Textbook with lots of information about language processing
http://www.cse.iitk.ac.in/users/mohit/Speech-and-Language-Processing.pdf

//Conditional random fields appears to be a common choice for machine learned NER:
http://en.wikipedia.org/wiki/Conditional_random_field

// A few articles on previous implementations
http://cogprints.org/5025/1/NRC-48727.pdf
  "2.1.1 Retrieve Pages with Seed 
  The first step is information retrieval from the Web. A query is created by conjoining a 
  seed of k manually generated entities (e.g., “Montreal” AND “Boston” AND “Paris” AND 
  “Mexico City”). In our experience, when k is set to 4 (as suggested by Etzioni et al. [4]) 
  and the seed entities are common city names, the query typically retrieves Web pages that 
  contain many names of cities, in addition to the seed names. The basic idea of the 
  algorithm is to extract these additional city names from each retrieved Web page. 
  The same strategy can be applied to person names, company names, car brands, and 
  many other types of entities. Although it is outside of the scope of this paper, we should 
  mention that we successfully applied this technique to more than 50 named-entity types. 
  2.1.2 Apply Web Page Wrapper 
  A Web page wrapper is a rule-based system that identifies the location of specific types of 
  information within a Web page. For example, a wrapper for identifying the location of 
  news headers on the Web site radio-canada.ca might contain the rule, “A header is an 
  HTML node of type <a>, with text length between 10 and 30 characters, in a table of 
  depth 5 and with at least 3 other nodes in the page that satisfy the same rule.” 
  The gazetteer generation algorithm proceeds by learning rules that identify the 
  locations of positive examples. For each page found in 2.1.1, a Web page wrapper is 
  trained on the k positive examples that are known to appear in the page, but only if they 
  are strictly contained in an HTML node (e.g., <td> Boston </td>) or surrounded by a 
  small amount of text inside an HTML node (e.g., <td> Boston hotel </td>). The remaining 
  HTML nodes in the page are treated as if they were negative examples, but we only 
  include in the negative set the nodes with the same HTML tags as the positive examples 
  [11]. For instance, if the k positive nodes are tagged as bold (i.e., “<b>”), then the 
  negative examples will be restricted to the remaining bold text in the Web page. The Web 
  page wrapper we used is similar to Cohen and Fan’s [2] wrapper, in terms of the learning 
  algorithm and the feature vector. "

http://cogprints.org/5859/1/Thesis-David-Nadeau.pdf
http://acl.ldc.upenn.edu/J/J01/J01-1005.pdf
http://www.dfki.de/~neumann/esslli04/reader/ie-lec3-1.pdf
http://machinelearningtext.pbworks.com/w/file/fetch/48758502/algthatlearns-Bikel.doc.pdf

//An interesting application: NER on tweets. Something similar might be a good hook
https://homes.cs.washington.edu/~mausam/papers/emnlp11.pdf

//possible availiable corpii for use in project:

http://www.cs.technion.ac.il/~gabr/resources/data/ne_datasets.html

//Perhaps a further narrowing of focus? A paper about the automatic creation of Gazetteers (usually created by hand):
http://aclweb.org/anthology//P/P08/P08-1047.pdf
