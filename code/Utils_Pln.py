#!/usr/bin/env python
# -*- coding: utf-8 -*-
from _pickle import dump, load
import _pickle
import codecs
import os

import pprint  # For proper print of sequences.
import re, string
import sys
import time
from unicodedata import normalize
import unicodedata
from xml.dom import minidom
import nltk
from nltk import BigramTagger as bt
from nltk import UnigramTagger as ut
import nltk, nltk.metrics
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy, math, unicodedata, re
# import treetaggerwrapper


import spacy
from spacy import displacy
# from tabulate import tabulate
import pandas as pd

# local_nlp =spacy.load('en')
local_nlp = spacy.load('en_core_web_lg')

import nltk.data

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import tags_map


class UtilPln():
    _filePathOntologia = ""
    _urlOntologia = 'http://localhost:8080/ReformulaPregunta/ReformularPreguntaService?WSDL'
    _namespaceOntologia = 'http://servicioweb/'
    _document = ""
    _collectionDocument = ""
    _KeyProcessFreqDist = []
    _features_doc = []
    _QueryProcess = []
    _ewnRelation = []
    _ewnSynset = []
    _ewnVariant = []

    def __init__(self):

        # StopWord
        #
        self.nlp = None

        self.Conditional_patterns = []

        docBlank = local_nlp(' ')

        self.tokenBlank = docBlank[0]

        self.NameTagsAHohfeld = tags_map.tag_Ahohfeld(self)

        # self.Conditional_patterns.append([
        #                               ['SW',  ["ADP","ADV"],       ["mark", "advmod", "prep"]], #"                           #,"acomp","acl","csubj","ccomp"]   ],
        #                               ['VBC', ["VERB"],            ["advcl","ccomp","csubjpass","xcomp","auxpass","conj"]], #
        #                               ['MV',  ["VERB"],            ["ROOT","MainVerb","ccomp","advcl","acl","pcomp","relcl","conj","preconj","xcomp"]],
        #                               ['1']
        #                              ])
        # self.Conditional_patterns.append([
        #                               ['VBC',["VERB"],            ["advcl","ccomp","acl","csubj","prep"]], #pcomp",
        #                               ['SW', ["ADP","ADV","VERB"],["advmod","prep", "ccomp", "acomp","csubj","advcl"]], #"pcomp" ,"mark"
        #                               ['MV', ["VERB"],            ["ROOT", "MainVerb","ccomp","xcomp","advcl","pcomp","conj"] ], # "acl" "conj" "acomp",
        #                               ['2'] ])

        ##self._stopWords = nltk.corpus.stopwords.words('spanish')

    def comparePattern(self, pattern, Conditional_Pattern):

        lenPattern = [len(Conditional_Pattern), "*"] if (len(pattern) > len(Conditional_Pattern)) else [len(pattern),
                                                                                                        ""]

        for i in range(0, lenPattern[0]):
            if pattern[i][0] not in Conditional_Pattern[i][0] or pattern[i][1] not in Conditional_Pattern[i][1]:
                return False, lenPattern[1]

        return True, lenPattern[1]

    def Detect_Pattern_old(self, token_SW, token_VBC, tokens_MV):

        token_MV_pattern = self.tokenBlank
        pattern = []
        # look forward front SW
        for t in token_SW.children:

            if t == token_VBC:
                pattern.append(['VBC', t.pos_, t.dep_, t.text])

                break

        # add SW
        pattern.append(['SW', token_SW.pos_, token_SW.dep_, token_SW.text])

        # look down front SW
        for t in token_SW.ancestors:

            if t == token_VBC:
                pattern.append(['VBC', t.pos_, t.dep_, t.text])
            elif t in tokens_MV:  # finish pattern
                token_MV_pattern = t
                pattern.append(['MV', t.pos_, t.dep_, t.text])
                break
            else:
                pattern.append(['XX', t.pos_, t.dep_, t.text])

                # compare pattern with conditional_pattern

        for p in self.Conditional_patterns:
            lenPattern = [len(p), "*"] if (len(pattern) > len(p)) else [len(pattern), ""]

            if len(pattern) < len(p) - 1:  # last pos is 'name pattern'
                return False, pattern, token_MV_pattern, 'X'

            pattern_ok = True

            for i in range(0, lenPattern[0] - 1):  # last pos is 'name pattern'
                if pattern[i][1] not in p[i][1] or pattern[i][2] not in p[i][2]:
                    pattern_ok = False

            if pattern_ok:
                return True, pattern, token_MV_pattern, p[3] + lenPattern

        return False, pattern, token_MV_pattern, 'X'

    def ToVectorNormalizado(self, document, keysCorpus, Alldocument):

        document_words = set(document)  # words del documento
        frec_doc = nltk.FreqDist(document)
        features = []

        for word in keysCorpus:  # Word_features -Todos los documentos
            if word in document_words:
                try:
                    tf = float(frec_doc[word]) / float(len(document))  # frec word en el documento
                    idf = math.log(float(len(Alldocument)) / float(
                        self.FrecWordDocument(word, Alldocument)))  # frec word en todos documento
                    tf_idf = tf * idf
                    features.append(tf_idf)
                except ZeroDivisionError:
                    features.append(0)
            else:

                features.append(0)

        return features

    def FrecWordDocument(self, word, doc_Preprocess):

        iveces = 0
        for doc in doc_Preprocess:
            for wd in doc:
                if word in wd:
                    iveces = iveces + 1
                    break
        return iveces

    def Pre_Process2(self, lin):
        lin = unicodedata(lin)  # Transformacion a unicode.

        lin = self.DeleteSpecialCharacters(lin)

        lin = self.DeleteAccents(lin)

        lin = nltk.word_tokenize(lin)
        lin3 = []
        lin2 = []

        for t in lin: lin2.append(self._spanishStemmer.stem(t))
        for t in lin2:
            if (t not in self._stopWords2):
                lin3.append(t)

        lin_preprocess = " ".join(lin3)

        return lin_preprocess

    def PreProcesador2(self, texto):

        return " ".join(self.Pre_Process3(texto))

    def Pre_Process3(self, lin):
        global t0
        t0 = time.time()
        # lin= unicode(lin) #Transformacion a unicode.
        lin = self.DeleteAccents(lin)
        lin1 = self.DeleteSpecialCharacters(lin)
        lin2 = nltk.word_tokenize(lin1)

        lin3 = []
        lin4 = []
        # TO DO --Debe soportar mas de un idioma
        self._Stemmer = SnowballStemmer('english')

        for w in lin2: lin3.append(self._Stemmer.stem(w))
        for w in lin3:
            if (w not in self._stopWords2) and len(w) > 2:
                lin4.append(w)

        return lin4  # timeDuration

    def PreProcesador(self, texto):

        return " ".join(self.PreProcess(texto), 'en')

    def PreProcess(self, lin, lang="en"):
        global t0
        t0 = time.time()
        # lin= unicodedata(lin) #Transformacion a unicode.
        lin = self.DeleteAccents(lin)
        lin1 = self.DeleteSpecialCharacters(lin)
        lin2 = nltk.word_tokenize(lin1)

        lin3 = []
        lin4 = []

        for w in lin2: lin3.append(self.Stemmer(w, lang))
        for w in lin3:
            if (w not in self._stopWords2) and len(w) > 2:
                lin4.append(w)

        return lin4  # timeDuration

    def Stemmer(self, word, lang):
        if (lang == "es"):
            language = 'spanish'
        if (lang == "en"):
            language = "english"

        self._Stemmer = SnowballStemmer(language)

        return self._Stemmer.stem(word)

    def DeleteSpecialCharacters(self, lin):
        lin = re.sub('\/|\\|\\.|\,|\;|\:|\n|\?|\¿|\¡|\!|\'|\)|\(', ' ', lin)  # quita los puntos
        lin = re.sub("\s+\w\s+", " ", lin)  # quita los caractores solos
        lin = re.sub("[0-9]", "", lin)  # quita los numeros
        lin = re.sub("\.", "", lin)
        lin = re.sub("\�", "", lin)
        lin = re.sub("\?", "", lin)
        return lin.lower()

    def DeleteAccents(self, _word):
        return ''.join((c for c in unicodedata.normalize('NFD', _word) if unicodedata.category(c) != 'Mn'))

    def DeleteSpecialCharacters2(self, lin):
        pattern = re.compile('\W')
        lin = re.sub(pattern, ' ', lin)

        # lin = re.sub('\/|\\|\\.|\,|\;|\:|\n|\?|\�|\*', ' ',lin) # quita los puntos
        # lin = re.sub("\s+\w\s+"," ",lin ) # quita los caractores solos
        # lin = re.sub("\.","",lin)
        return lin.lower()

    def DeleteStopWord(self, documento):
        d1 = []
        for word in documento:
            if (word.lower() not in stopwords) and (word != "") and (len(word) > 2):
                d1.append(word.lower())
        return d1

    def CheckFile(self, archivo):
        import os.path
        if os.path.exists(archivo):
            return True
        else:
            return False

    def POStagger(self, text, lang):
        _tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)
        tags = _tagger.tag_text(text)
        # pprint.pprint(tags)

        return tags

    def ExtraccionTerminosClaves(self, oQuestion):

        catGramaticales = oQuestion.posTagger
        keyWords = []
        for cat in catGramaticales:
            if cat[1] == "N" or cat[1] == "A" or cat[1] == None or cat[1] == "V":
                keyWords.append(cat[0])

        print("Extraccion Terminos Claves" + str(keyWords))

        return keyWords

    def SimilitudCoseno(self, u, v):
        return numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))

    def ExpandirTerminosConEuroWordnet(self, terminos):
        print("ExpandirTerminosConEuroWordnet terminos :" + " ".join(terminos))
        _terminosExpandidos = []
        for termino in terminos:
            print("ExpandirTerminosConEuroWordnet  termino:" + termino)
            sinonimos = self.ewn_TypeRelationsWord(termino, 'near_synonym')
            for sinonimo in sinonimos:
                _terminosExpandidos.append(sinonimo)
        # print _terminosExpandidos
        y = self.EliminaDuplicadosEnLista(_terminosExpandidos)

        return _terminosExpandidos

    def EliminaDuplicadosEnLista(self, lista):
        lst2 = []

        for key in lista:
            if key not in lst2:
                lst2.append(key)
        return lst2

    def ExpansionConOntologia(self, termino):

        print("ExpansionConOntologia :" + termino)

        # 1. Defino un arreglo de las PreguntaExpandida en nulo
        VectorPreguntaExpandida = []
        palabrasEncontradasEnOntologia = []

        # busco palabra en ontologia
        for palabra in pregunta.split(" "):
            palabrasEncontradasEnOntologia = self.BuscarPalabraEnOntologia(palabra)
            # si hay por lo menos una clase reemplazar
            if palabrasEncontradasEnOntologia != 'null':
                for clase in palabrasEncontradasEnOntologia:
                    # si la encuentro reemplazo palabra en la pregunta y agrego a PreguntaExpandida
                    if clase != 'null' and clase != '':
                        # cambia clase de ontologia por palabra de la pregunta
                        preguntaExpandida = pregunta.replace(palabra, clase)
                        VectorPreguntaExpandida.append(preguntaExpandida)

        return VectorPreguntaExpandida

    def ExpandirTerminosCiudadano(self, terminosCiudadano_p):
        terminosGobierno = []
        for terminoCiudadano in terminosCiudadano_p:
            for termino in self.terminosDominio:
                if termino[0] == terminoCiudadano:
                    terminosGobierno.append(termino[3])

        return terminosGobierno

    ##     Retorna los terminos relacionados que utiliza el ciudadano
    def ExpandirTerminosGobierno(self, terminoGobierno):
        terminoCiudadano = []
        for termino in self.mapeo:
            if terminoGobierno in termino.split(';')[1]:
                terminoCiudadano.append(termino.split(';')[0])

        self.EliminarRepetidas(terminoCiudadano)

    # buscar palabras en ontologia
    def BuscarPalabraEnOntologia(clase):
        url = 'http://localhost:8080/ReformulaPregunta/ReformularPreguntaService?WSDL'
        namespace = 'http://servicioweb/'
        server = SOAPProxy(url, namespace)
        # nueva = server.clase_SubClases(Clase = clase)
        nueva = server.subClase_Clase(subClase=clase)
        return nueva

    def Read_Config(self, xmlTag):
        return self.Read_XML("Config.XML", xmlTag)

    def Read_XML(self, xmlFile, xmlTag):
        resultList = []
        try:
            dom = minidom.parse(xmlFile)
            elements = dom.getElementsByTagName(xmlTag)
            if len(elements) != 0:
                for i in range(0, len(elements)):
                    resultList.extend([elements[i].childNodes[0].nodeValue])
            else:
                print('xxx No hay elementos en el fichero XML con la etiqueta ' + xmlTag)
        except:
            print(os.getcwd())
            print('xxx El fichero no existe o esta mal formado.')
            print('xxx Path del fichero: ' + xmlFile)
            print('xxx Etiqueta sobre la que se realiza la busqueda: ' + xmlTag)
        return resultList

    def Read_XML_Atribute(self, xmlFile, xmlTag, xmlAtribute):
        resultList = []
        try:
            dom = minidom.parse(xmlFile)
            elements = dom.getElementsByTagName(xmlTag)
            if len(elements) != 0:
                for i in range(0, len(elements)):
                    resultList.extend([elements[i].attributes.getNamedItem(xmlAtribute).childNodes[0].nodeValue])
            else:
                print('xxx No hay elementos en el fichero XML con la etiqueta ' + xmlTag)
        except:
            print('xxx El fichero no existe o esta mal formado.')
            print('xxx Path del fichero: ' + xmlFile)
            print('xxx Etiqueta sobre la que se realiza la busqueda: ' + xmlTag)
        return resultList

    def Read_XML_File(self, xmlFile, nameNode, aNameElements):
        resultList = []
        dom = minidom.parse(xmlFile)
        elements = dom.getElementsByTagName(nameNode)
        # print(dom)
        # print(elements)
        for e in elements:
            a = []
            for n in aNameElements:
                # print (n)
                x = e.getElementsByTagName(n)[0]

                y = x.childNodes[0].nodeValue
                print(y)
                a.append(y)
            resultList.append(a)
        return resultList

    ##    -----------
    def ewn_ReadFiles(self):
        # carga y lee Relation
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-relation")
        renglones = f.readlines()
        relation = map(lambda x: x.split("|"), renglones)
        # carga y lee Synset
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-synset")
        renglones = f.readlines()
        synset = map(lambda x: x.split("|"), renglones)
        # carga y lee Variant
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-variant")
        renglones = f.readlines()
        variant = map(lambda x: x.split("|"), renglones)
        # retorna los archivos
        return relation, synset, variant
        # lee y carga el archivo RELATION de WordNet

    def ewn_ReadRelation(self):
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-relation")
        renglones = f.readlines()
        relation = map(lambda x: x.split("|"), renglones)
        return relation

    # lee y carga el archivo SYNSET de WordNet
    def ewn_ReadSynset(self):
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-synset")
        renglones = f.readlines()
        synset = map(lambda x: x.split("|"), renglones)
        return
        # lee y carga el archivo VARIANT de WordNet

    def ewn_ReadVariant(self):
        f = open(self._FilePath_EuroWordNet + "\esWN-200611-variant")
        renglones = f.readlines()
        variant = map(lambda x: x.split("|"), renglones)
        return variant

        # Busqueda de la palabra en el archivo Variant

    ##    self._ewnRelation,self._ewnSynset,self._ewnVariant
    def ewn_offsetsWord(self, palabra):

        offsets = []
        for variant in self._ewnVariant:
            if palabra in variant:
                # guarda los offset que encuentra de la palabra
                offsets.append(variant[1])
        # si el arreglo esta vacio es por que la palabra no existe

        return offsets

    def ewn_WordsOffset(self, offset):

        words = []
        for variant in self._ewnVariant:
            if offset in variant:
                words.append(variant[2])

        return words

    def ewn_TypeRelationsWord(self, palabra, tipo):
        # llama al metodo para buscar la palabra
        offsets = self.ewn_offsetsWord(palabra)

        # carga el archivo Relation
        offsetRelations = []
        wordsRelations = ""
        for offset in offsets:
            # corre el arreglo offset para comparar los offset que
            # tiene y buscarlos en el archivo Relation
            for relation in self._ewnRelation:
                if relation[0] == tipo:
                    if (offset in relation[2]):
                        offsetRelations.append(relation[4])
                    if (offset in relation[4]):
                        offsetRelations.append(relation[2])
        for offset in offsetRelations:
            wordsRelations += " ".join(self.ewn_WordsOffset(offset)) + " "
        wordsRelations = wordsRelations.split()

        # print wordsRelations

        return wordsRelations

    def ewn_RelationsWord(self, palabra):
        # llama al metodo para buscar la palabra
        offsets = self.ewn_offsetsWord(palabra)

        # carga el archivo Relation
        offsetRelations = []
        wordsRelations = []
        for offset in offsets:
            # corre el arreglo offset para comparar los offset que
            # tiene y buscarlos en el archivo Relation
            for relation in self._ewnRelation:

                if (offset in relation[2]):
                    offsetRelations.append(relation[4])
                if (offset in relation[4]):
                    offsetRelations.append(relation[2])
        for offset in offsetRelations:
            wordsRelations.append(self.ewn_WordsOffset(offset))

        return wordsRelations

    def lexical_diversity(self, text):
        return len(text) / len(set(text))

    def ExpandeTerminos(self, oQuestion):
        _TermsExpandOntologia = []
        _TermExpandedEuroWordNet = []
        _TermExpandedDominio = []
        _keyTerms = oQuestion.keyWords
        _keyTerms_p = self.Pre_Process(_keyTerms)

        print("_keyTerms:")
        print(_keyTerms)
        print("_keyTerms_p")
        print(_keyTerms_p)

        if self._ExpandirConOntologia == "true":
            _TermsExpandOntologia = self.ExpandirTerminosConOntologia(_keyTerms)

        if self._ExpandirConEuroWordNet == "true":
            _TermExpandedEuroWordNet = self.ExpandirTerminosConEuroWordnet(_keyTerms)

        if self._ExpandirConTerminoDominio == "true":
            _TermExpandedDominio = self.ExpandirTerminosCiudadano(_keyTerms_p)

        self._TermExpanded = _keyTerms + _TermsExpandOntologia + _TermExpandedEuroWordNet + _TermExpandedDominio

        self._TermExpanded = self.EliminaDuplicadosEnLista(self._TermExpanded)

        return self._TermExpanded

    def remove_punctuation(self, text):
        return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    def getMultiWordTerms(self, taggedText):
        """Extracts terms composed of multiple words.
        e.g: 'sistema operativo', 'base de datos'
        @param taggedText: list of triples [word, POS, lemma]
        @return multiwordTerms: list of triples [multiword_term, 'TERM', '<unknown>']
        @return indices: indices of multiwordTerms in taggedText
        Patterns: (N=nombre, A=adjetivo, P=preposición, V=verbo, PCLE=partícula)
        N-A; A-N; N-N; N-P-N; V-PCLE
        ref: http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish-tagset.txt
        """
        multiwordTerms = []
        indices = []  # WordIndices
        adjetivo = ['JJ']
        nombre = ['NN', 'NNS', 'NP', 'NPS']
        preposicion = ['PP', 'DT']
        n = len(taggedText)

        for i in range(n - 1):
            # BIGRAM
            if (i + 1) < n:
                bigram = ""
                # Get 1st and 2nd tagged words

                taggedWord1 = taggedText[i].split()
                taggedWord2 = taggedText[i + 1].split()

                # Get POS tags and words
                pos1 = taggedWord1[1]
                pos2 = taggedWord2[1]

                word1 = taggedWord1[0]
                word2 = taggedWord2[0]

                # Case 1: N-A
                case1 = (pos1 in nombre and pos2 in adjetivo)
                # Case 2: A-N
                case2 = (pos1 in adjetivo and pos2 in nombre)
                # Case 3: N-N
                case3 = (pos1 in nombre and pos2 in nombre)

                isBigram = case1 or case2 or case3
                if isBigram:
                    bigram = word1 + ' ' + word2
                    taggedTerm = [bigram, 'TERM', u'<unknown>']
                    multiwordTerms.append(taggedTerm)
                    indices.append(i)  # WordIndices
                    indices.append(i + 1)  # WordIndices

                # TRIGRAM
                if (i + 2) < n:
                    trigram = ""
                    # Get 3rd tagged word
                    taggedWord3 = taggedText[i + 2].split
                    # Get POS tag and word
                    pos3 = taggedWord3[1]
                    word3 = taggedWord3[0]

                    # Case 4: N-P-N
                    case4 = (pos1 in nombre and pos2 in preposicion and pos3 in nombre)

                    isTrigram = case4
                    if isTrigram:
                        trigram = word1 + ' ' + word2 + ' ' + word3
                        taggedTerm = [trigram, 'TERM', u'<unknown>']
                        multiwordTerms.append(taggedTerm)
                        indices.append(i)  # WordIndices
                        indices.append(i + 2)  # WordIndices

        return multiwordTerms, indices  # WordIndices

    def leaves(self, tree, filter):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter=lambda t: t.label() == filter):
            yield subtree.leaves()

    def foundTagsAhohfeld(self, string, parameters=False):

        tagsText = []
        tagsHoh = []

        ns = string.strip()
        ln = len(ns)
        f = ns.find(")")
        i = ns.find("(")
        d = 0
        while f < ln:
            param = ns[i + 1:f].split(";")
            tag = ns[d:i]
            if parameters == True:
                if tag.strip().upper() in self.NameTagsAHohfeld:
                    tagsHoh.append([tag, param])
            else:
                tagsText.append(tag)

            d = f + 1
            i = d + ns[d:ln].find("(")
            f = d + ns[d:ln].find(")") if (ns[d:ln].find(")") > 0) else ln

        if parameters == False:
            for tag in tagsText:
                for t in tag.split():
                    if t.upper() in self.NameTagsAHohfeld:
                        if t.upper() not in tagsHoh:
                            tagsHoh.append(t.upper())

        return tagsHoh

    def explainTagAhohfeld(self, tag):
        return self.NameTagsAHohfeld[tag]

    def count_Categoria(self, dic, value):
        if value in dic:
            val = dic[value] + 1
            dic[value] = val
        else:
            dic[value] = 1

    def old_readGold(self):
        self._NameFileNorms = self.Read_Config('FilePath_Norms')[0]

        norms = []
        # 0 - < NORM_ID >
        # 1 - < NORM_TEXT >
        # 2 - < NORM_LANG >
        # 3 - < NORM_URL >
        # 4 - < NORM_COMMUNITY >
        # 5 - < NORM_EFFECT_BY_VIOLATION >
        # 6 - < NORM_EFFECT_BY_COMPLIANCE >
        # 7 - < AHOHFELD_TAG >

        norms = self.Read_XML_File(self._NameFileNorms, "NORM",
                                   ['ID_TEXT', 'TEXT', 'LANGUAGE', 'TAG_PREDICATE', 'TAG_SUBJECT', 'TAG_OBJECT',
                                    'TAG_LEGAL_AHOHFELD', 'NORM_URL', 'NORM_COMMUNITY', 'CONTRACT_SECTION',
                                    'NORM_NAME_SECTION', 'NORM_NUMBER_SECTION', 'COMMENTS'])

        return norms

    def found_value_parameter(self, nameParameter, array, Categoria=False):
        # array [ categoria , [ parameter1= Value1, parameter2= value2, ..parametern= valuen]
        # supossed one record
        value = ""
        parameters = array[0][1]
        for p in parameters:
            x = p.split("=")
            name = x[0]
            if name.strip().lower() == nameParameter.strip().lower():
                value = x[1].strip().lower()
                return value

        return value

    def found_value_tag(self, nameTag, array):
        # array [ categoria , [ parameter1= Value1, parameter2= value2, ..parametern= valuen]
        value = array[0][0]
        tag = array[0][0]
        for t in array:
            if t[0].strip().upper() == nameTag.strip().upper():
                return t[0].strip().upper()
        return value

    class Bunch(dict):
        """Container object for datasets

        Dictionary-like object that exposes its keys as attributes.


        """

        def __init__(self, **kwargs):
            super(Bunch, self).__init__(kwargs)

        def __setattr__(self, key, value):
            self[key] = value

        def __dir__(self):
            return self.keys()

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setstate__(self, state):
            # Bunch pickles generated with scikit-learn 0.16.* have an non
            # empty __dict__. This causes a surprising behaviour when
            # loading these pickles scikit-learn 0.17: reading bunch.key
            # uses __dict__ but assigning to bunch.key use __setattr__ and
            # only changes bunch['key']. More details can be found at:
            # https://github.com/scikit-learn/scikit-learn/issues/6196.
            # Overriding __setstate__ to be a noop has the effect of
            # ignoring the pickled __dict__
            pass

    def lookMoreOtherVerbs(self, token_verb, otherVerbs, tags):

        for token in token_verb.children:
            if token.dep_ in tags and token.pos_ in ('VERB'):  # and token != token_verb:
                otherVerbs.append([token, self.foundAuxVerb(token), '', ''])
                self.lookMoreOtherVerbs(token, otherVerbs, tags)

        return otherVerbs

    def foundOtherVerbs(self, mainVerbs):
        otherVerbs = []

        for i in range(0, len(mainVerbs)):  # add other verbs coneccte with main verb with other dep
            verb = mainVerbs[i][0]

            aditionalVerb = None;
            aux = "";
            auxPass = ""
            neg = ""

            tags = self.tags_Connected_OtherVerbs

            for t in verb.children:
                if t.dep_ in tags and t.pos_ == "VERB":
                    aditionalVerb = t
                    otherVerbs.append([aditionalVerb, self.foundAuxVerb(aditionalVerb), '', ''])
                    self.lookMoreOtherVerbs(aditionalVerb, otherVerbs, tags)

        return otherVerbs

    def deprecated_foundAuxVerb(self, token_verb):
        aux = "";
        auxPass = "";
        neg = ""
        for t in token_verb.children:
            if t.dep_ in ('aux'):
                aux += t.text_with_ws
            elif t.dep_ in ('auxpass'):
                auxPass += t.text_with_ws
            elif t.dep_ in ('neg'):
                neg = t.text_with_ws
        return [aux, neg, auxPass, token_verb.text]

    def aux_mainVerbs(self, tokens_mainVerbs_withOutAux):

        tokens_mainVerbs = []
        for i in range(0, len(tokens_mainVerbs_withOutAux)):
            auxs = self.foundAuxVerb(tokens_mainVerbs_withOutAux[i])
            tokens_mainVerbs.append([tokens_mainVerbs_withOutAux[i], auxs[0], auxs[1], auxs[2], auxs[3]])

        for i in range(0, len(tokens_mainVerbs)):
            if (tokens_mainVerbs[i][1] == self.tokenBlank):
                for j in range(i - 1, -1, -1):
                    if tokens_mainVerbs[j][1] != self.tokenBlank:
                        tokens_mainVerbs[i][1] = tokens_mainVerbs[j][1]

            if tokens_mainVerbs[i][3] == self.tokenBlank:
                for j in range(i - 1, -1, -1):
                    if tokens_mainVerbs[j][3] != self.tokenBlank:
                        tokens_mainVerbs[i][3] = tokens_mainVerbs[j][3]

        return tokens_mainVerbs

    def SOintroduced_by_SW(self, token_verb, token_SW, token_sentence):

        if token_SW.nbor().dep_ in ('nsubj', 'nsubjpass'):  # 'expl'
            return True

        n = []
        for chunk in token_sentence:
            n.append([chunk.text, chunk.dep_])

        # for t in verb.children:
        #    if t.text.lower()==SW.text.lower() and t.nbor().dep_ in ('poss','nsubj'):#'expl'
        #        return True
        return False

    def IsClause(self, text_Clause):
        doc_Clause = self.nlp(text_Clause)

        existVerb = False
        existSubject = False

        if len(doc_Clause) < 3:
            return '', False

        txtClause = ""
        for i in range(0, len(doc_Clause)):
            token = doc_Clause[i]
            nextToken = doc_Clause[i]

            if i + 1 < len(doc_Clause):
                nextToken = doc_Clause[i + 1]

            if token.text.lower() in (
            ",", ";", "then") and existVerb and existSubject:  # ,"ADJ"): and  nextToken.pos_ in ("DET","NOUN") )
                break

            if token.pos_ == "VERB":
                existVerb = True
            if token.dep_ in ("nsubj", "csubj", "csubjpass", "nsubjpass", "exp"):
                existSubject = True

            txtClause += token.text_with_ws

        # TODO finsih check isClause

        return txtClause, True  # (existVerb and existSubject)

    def foundConditionalbyPatterns(self, token_mainVerb, token_signalWord):

        Pattern1 = [[['prep', 'advcl', 'ccomp', 'advmod'], ['ADP']]]

        Pattern2 = [[['ccomp', 'acomp', 'xcomp'], ['VERB']], [['advcl']]]

    def old_foundTenseVerb(self, array_tokenVerb):

        tokenVerb = array_tokenVerb[0]

        if tokenVerb == self.tokenBlank:
            return "withoutVerb"

        token_aux_modal = array_tokenVerb[1]
        token_aux = array_tokenVerb[2]
        token_auxpass = array_tokenVerb[3]

        auxVerb = ""
        for i in range(1, 4):
            if array_tokenVerb[i] != self.tokenBlank:
                auxVerb += array_tokenVerb[i].text + " "
        auxVerb = auxVerb.strip().lower()

        # auxVerb = (token_aux.text + " " + (token_aux2.text+" ").strip()+token_auxpass.text).strip().lower()

        # VB	VERB	VerbForm=inf	verb, base form
        # VBD	VERB	VerbForm=fin 	Tense=past	verb, past tense
        # VBG	VERB	VerbForm=part 	Tense=pres Aspect=prog	verb, gerund or present participle
        # VBN	VERB	VerbForm=part 	Tense=past Aspect=perf	verb, past participle
        # VBP	VERB	VerbForm=fin 	Tense=pres verb, non-3rd person singular present
        # VBZ	VERB	VerbForm=fin 	Tense=pres Number=sing Person=3	verb, 3rd person singular present

        auxPerfectConditional = ["would have", "could have"]
        auxPastPerfect = ['had']
        auxPresentPerfect = ['has', 'have']
        auxPerfectContinousConditional = ["would have been", "could have been"]
        auxPresentContinousConditional = ["would be", "could be"]
        auxPresentConditional = ["would", "could", "ought"]
        auxFuture = ["will", "will not", "won't", "shall",
                     "will be"]  # "shall","will be"] # ,"shall","can","could","may","might","should"]
        auxPast = ["did"]
        auxInfinitive = ["PART"]

        verbBaseForm = ['VB']
        verbPresentParticiple = ['VBG']
        verbPastSimple = ['VBD', 'VBN']
        verbPastParticiple = ['VBN']
        verbSimplePresent = ['VBP', 'VBZ', 'VB']

        tense = ""

        # nlp.vocab.morphology.tag_map

        if tokenVerb.tag_ in verbPresentParticiple and auxVerb == "":
            return "presentParticiple"

        # El "present conditional" de cualquier verbo está compuesto por dos elementos:
        # "would" + infinitfivo sin "to" del verbo principal
        if (auxVerb in auxPresentConditional and tokenVerb.tag_ in verbBaseForm) or (
                token_aux_modal.text in auxPresentConditional and token_auxpass.tag_ in verbSimplePresent):
            tense = "presentConditional"

        # The present continuous conditional tense of any verb is composed of three elements:
        # would + be + present participle
        elif auxVerb in auxPresentContinousConditional and tokenVerb.tag_ in verbPresentParticiple:
            tense = "presentContinuousConditional"

        # El "perfect conditional" de cualquier verbo está compuesto por tres elementos:
        # would + have + past participle
        elif auxVerb in auxPerfectConditional and tokenVerb.tag_ in verbPastParticiple:
            tense = "perfectConditional"

        #  El perfect continuous conditional de cualquier verbo está compuesto por cuatro elementos:
        # would + have + been + present  participle
        elif auxVerb in auxPerfectContinousConditional and tokenVerb.tag_ in verbPresentParticiple:
            tense = "perfectContinousConditional"

        # El "present perfect" está compuesto por dos elementos: forma apropiada del verbo auxiliar to have (en presente) y el "past participle" del verbo principal.
        elif auxVerb in auxPresentPerfect and tokenVerb.tag_ in verbPastParticiple:
            tense = "PresentPerfect"


        # El "simple future" está compuesto por dos partes: will / shall + infinitivo sin to

        elif auxVerb.lower() in auxFuture and (tokenVerb.tag_ in verbBaseForm or token_auxpass.tag_ in verbBaseForm):
            tense = "simpleFuture"

        elif (
                tokenVerb.tag_ in verbSimplePresent or token_auxpass.tag_ in verbSimplePresent or token_aux.tag_ in verbSimplePresent or token_aux_modal.tag_ in verbSimplePresent) and auxVerb not in auxPast:
            tense = "simplePresent"  # or  token_aux_modal.tag_ in verbSimplePresent

        elif tokenVerb.tag_ in verbPastSimple or token_auxpass.tag_ in verbPastSimple or auxVerb in auxPast:
            tense = "pastSimple"

        # El "past perfect" está compuesto por : el pasado del verbo to have (had) + el "past participle" del verbo principal.
        elif tokenVerb.tag_ in verbPastParticiple and auxVerb in auxPastPerfect:
            tense = "pastPerfect"

        if token_aux.pos_ in auxInfinitive:
            return "infinitive"

        return tense

    def foundVerbConditional(self, signalWordFound, tokens):
        tokenVerb = None
        auxVerb = ""

        # array_signalWord= signalWordFound.split(" ")

        for t in tokens:

            if t.text.strip().lower() in signalWordFound:  # array_signalWord: # try reach verb
                for a in t.ancestors:
                    if a.pos_ == "VERB":
                        tokenVerb = a
                        auxTokenVerb = self.foundAuxVerb(tokenVerb)
                        auxVerb = (auxTokenVerb[0] + " " + auxTokenVerb[2]).strip().lower()
                        return tokenVerb, auxVerb

        return tokenVerb, auxVerb

    def foundVerbIn(self, Xtree, tokenMainVerb):

        tokenVerb = None
        auxVerb = ""

        for token in Xtree:
            if token.pos_ == "VERB" and tokenVerb == None and token.dep_ not in ("aux", "auxpass"):
                tokenVerb = token
            if token.dep_ in ("aux", "auxpass"):
                auxVerb += token.text.lower() + " "
        if tokenVerb == None:
            tokenVerb = tokenMainVerb

        return tokenVerb, auxVerb

    # def verbs_con_complemento(self, mainVerbs, otherVerbs):
    #
    #     for i in range(0, len(mainVerbs)):
    #         token_verb = mainVerbs[i][0]
    #         auxverb = mainVerbs[i][1]
    #         verb = mainVerbs[i][2]
    #         subject = mainVerbs[i][3]
    #         object = mainVerbs[i][4]
    #
    #         print("token_verb", token_verb)
    #         for t in token_verb.children:
    #             # print("t",t,"t_dep",t.dep_)
    #             if t.dep_ == "advcl":
    #                 for j in range(0, len(otherVerbs)):
    #                     if t == otherVerbs[j][0] and object == "" and otherVerbs[j][4] != "" and otherVerbs[j][3] == "":
    #                         mainVerbs[i][2] += " " + otherVerbs[j][1] + " " + otherVerbs[j][2]
    #                         mainVerbs[i][4] = otherVerbs[j][4]
    #                         print("t*", t)
    #

    # def found_MainVerbs(self, text_tokens):
    #     aux_verb = "";
    #     aux_neg = "";""
    #     object = "";
    #     subject = "";
    #     objectComplement = ""
    #     mainVerbs = []
    #
    #     for token in text_tokens:
    #         print(token.text, "\t\t" + token.dep_, "\t\t" + token.head.text, "\t\t" + token.head.pos_,
    #               [child for child in token.children])
    #
    #     for token in text_tokens:
    #
    #         if token.pos_ == "VERB":
    #             if token.dep_ == "ROOT":
    #                 initial = [token, '', '', '', '']
    #                 mainVerbs = self.found_verbs_dep(token, [initial])
    #
    #     return mainVerbs

    # def found_verbs_dep(self,token_verb, array):
    #
    #     verbs_dep_root = array
    #
    #     for token in token_verb.children:
    #         if token.dep_ in ('conj', "cc") and token.pos_ in ('VERB') and token != token_verb:
    #             # verbs_dep_root.append( [token,'','',''] )
    #             array.append([token, '', '', '', ''])
    #             self.found_verbs_dep(token, array)
    #
    #             # found_verbs_dep( token, verbs_dep_root )
    #
    #     return verbs_dep_root

    def clean_text(self, text):

        CARACTERES_A_BORRAR = ['\\n', '\n']

        for caracter in CARACTERES_A_BORRAR:
            texto = texto.replace(caracter, "")

        return texto

    def preProcessNorm(self, text_sentence):
        print("prepocesando")

        sentence = sent_detector.tokenize(text_sentence.strip())

        sentences = []
        for s in sentence:
            lines_sentence = s.split("\n")
            for line in lines_sentence:
                line_clean = re.sub(r'\([^)]*\)', '', line)  # quita los(1.1)
                line_clean = line_clean.split(".")
                for sc in line_clean:
                    if len(sc.strip()) > 10:
                        sentences.append(sc)

        conditionalSentences = []

        for s in sentences:
            signalWord = False
            s_clean = re.sub(r'[^)]*\)', '', s)  # quita los a)...
            s_clean = re.sub(r"\s+", " ", s_clean)

            s_clean = re.sub(r'[0-9]', '', s_clean)  # quita los numeros
            # s_clean= re.sub(r'\.', '', s_clean)  # quita los numeros
            s_clean = re.sub(r'\*', '', s_clean)  # quita los numeros
            # s_clean= re.sub("\s+\w\s+", "", s_clean)
            # s_clean=re.sub('\*+ ' , '', s_clean)  # quita los puntos

            if s_clean == s.upper():  # title
                continue

            s_clean = s_clean.strip()
            array_sentence_lower = "".join(s_clean).lower().split(" ")

            for cond in self.subOrdinanteAdverbialCondition:
                if cond in array_sentence_lower:
                    signalWord = True

            if signalWord and s_clean[0].upper() == s_clean[0]:
                conditionalSentences.append(s_clean)

        return conditionalSentences

    def posProcessNorms(self, clase, matriz_objects):
        new = sorted(matriz_objects, key=lambda clase: clase.TEXT)

        newListObject = []
        ant_Text = ""
        for o in new:
            norm_v = ""
            for v in o.tag_predicate:
                norm_v += (" ".join(v)).strip()
            if (o.TEXT != ant_Text) and (norm_v != ""):
                newListObject.append(o)
            ant_Text = o.TEXT

        return newListObject

    def found_Conditionals_Method_0(self, extendedVerbs, initialMainVerbs):

        conditionals = []

        conditions = []

        tokenMainVerb = initialMainVerbs[0][0]
        auxMainVerb = (initialMainVerbs[0][1][0] + " " + initialMainVerbs[0][1][2]).strip()

        tenseMainClause = self.foundTenseVerb(tokenMainVerb, auxMainVerb.lower())

        foundConditionInMainVerbs = False

        for i in range(0, len(initialMainVerbs)):

            conditions = self.foundSignalWordConditional(initialMainVerbs[i], 'MainVerb', tenseMainClause,
                                                         tokenMainVerb, auxMainVerb)
            # if len(conditions)==0:
            #    conditionals.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])
            if len(conditions) > 0:
                foundConditionInMainVerbs = True
                for condition in conditions:
                    conditionals.append(condition)

        # if not foundConditionInMainVerbs:
        #     for i in range(0, len(extendedVerbs)):
        #         conditions = self.foundSignalWordConditional(extendedVerbs[i], 'ExtendVerb',tenseMainClause,tokenMainVerb,auxMainVerb)
        #         if len(conditions) >0:
        #             for condition in conditions:
        #                 conditionals.append(condition)

        if len(conditionals) == 0:
            # Conditionals.append(['','','',tenseMainClause,aux_mainVerb,token_mainVerb.text,'','','','','','',''])
            conditionals.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])

        return conditionals

    def foundSignalWordConditional(self, svoVerb, typeVerb, tenseMainClause, tokenMainVerb, auxMainVerb):
        tokenVerb = svoVerb[0]

        Conditionals = []

        for token in tokenVerb.children:

            children_token = [w.text.lower() for w in token.children]
            subtree_token = [w.text.lower() for w in token.subtree]

            children_text = ''.join(w.text_with_ws.lower() for w in token.children)
            subtree_text = ''.join(w.text_with_ws.lower() for w in token.subtree)

            if token.dep_ in self.tags_Connected_ConditionalsWithVerbs:

                for signalWordCondition in self.subOrdinanteAdverbialCondition:

                    # wordsInSignalWordCondition= signalWordCondition.split(" ")
                    # if len(wordsInSignalWordCondition) == 1:
                    lsignalWordInToken = (signalWordCondition == token.text.lower().strip())
                    lsignalWordInSubtree = (signalWordCondition in subtree_token)
                    lsignalWordInChildren = (signalWordCondition in children_token)
                    # else:
                    #    lsignalWordInToken = False
                    #    lsignalWordInSubtree = (subtree_text.find(signalWordCondition) >=0)
                    #    lsignalWordInChildren = (children_text.find(signalWordCondition) >=0)

                    if lsignalWordInToken or lsignalWordInSubtree or lsignalWordInChildren:
                        Conditionals.append(self.ConditionalsInToken(signalWordCondition, token, tokenVerb, typeVerb,
                                                                     tokenMainVerb, auxMainVerb, svoVerb,
                                                                     tenseMainClause,
                                                                     lsignalWordInToken, lsignalWordInChildren,
                                                                     lsignalWordInSubtree))

        return Conditionals

    def ConditionalsInToken(self, signalWordFound, token, tokenVerb, typeVerb, tokenMainVerb, auxMainVerb, svoVerb,
                            tenseMainClause, lsignalWordInToken, lsignalWordInChildren, lsignalWordInSubtree):

        subject_mainVerb = svoVerb[2]
        object_mainVerb = svoVerb[3]

        txt_MainVerb = svoVerb[1][0] + " " + svoVerb[1][2] + " " + svoVerb[1][3]

        token_mainVerbInitial = []

        # token_ConditionalVerb = tokenVerb
        # auxTokenVerb = self.foundAuxVerb(tokenVerb)

        # aux_ConditionalVerb = (auxTokenVerb[0] + " " + auxTokenVerb[2]).strip().lower()

        signalWordinWord = "";
        signalWordInChildren = "";
        signalWordInSubtree = ""
        if lsignalWordInToken and lsignalWordInChildren == False and lsignalWordInSubtree == False:
            signalWordinWord = token.dep_
            # token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbIn(tokenVerb.subtree, tokenVerb)
            token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbConditional(signalWordFound, [token])

            # condition = "["+token_ConditionalVerb.text+"]"+''.join(w.text_with_ws for w in tokenVerb.subtree)

            condition = ''.join(w.text_with_ws for w in tokenVerb.subtree)

        elif lsignalWordInChildren:

            signalWordInChildren = token.dep_
            # token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbIn(token.children, tokenVerb)
            token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbConditional(signalWordFound, token.children)

            # condition = "["+token_ConditionalVerb.text+"]"+''.join(w.text_with_ws for w in token.subtree)
            condition = ''.join(w.text_with_ws for w in token.subtree)



        elif lsignalWordInSubtree:
            signalWordInSubtree = token.dep_
            # token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbIn(token.subtree, tokenVerb)
            token_ConditionalVerb, aux_ConditionalVerb = self.foundVerbConditional(signalWordFound, token.subtree)

            # condition = "["+token_ConditionalVerb.text+"]"+''.join(w.text_with_ws for w in token.subtree)

            condition = ''.join(w.text_with_ws for w in token.subtree)

        tenseConditionalClause = self.foundTenseVerb(token_ConditionalVerb, aux_ConditionalVerb)

        typeCondition = self.canonicalPattern(tenseMainClause, tenseConditionalClause)

        txt_ConditionVerb = aux_ConditionalVerb + " " + token_ConditionalVerb.text

        if token_ConditionalVerb == None:
            conditionVerb = ""
        else:
            conditionVerb = token_ConditionalVerb.text

        return [condition,
                signalWordFound,
                typeCondition,
                "[" + txt_MainVerb + "]" + tenseMainClause,
                auxMainVerb,
                tokenMainVerb.text,
                "[" + txt_ConditionVerb + "]" + tenseConditionalClause,
                aux_ConditionalVerb,
                conditionVerb,
                signalWordinWord,
                signalWordInChildren,
                signalWordInSubtree,
                typeVerb]
