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
  // useful information about scoring classifier performance...

//Short powerpoint going over the steps of an semi-supervised algorithm for NER:
http://www.cs.columbia.edu/~jebara/6772/proj/an2505_project.pdf

//Textbook with lots of information about language processing
http://www.cse.iitk.ac.in/users/mohit/Speech-and-Language-Processing.pdf

//Conditional random fields appears to be a common choice for machine learned NER:
http://en.wikipedia.org/wiki/Conditional_random_field

// A few articles on previous implementations
http://cogprints.org/5025/1/NRC-48727.pdf
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
