# -*- coding: utf-8 -*-
'''
Created on 10/04/2017
@author: Beltran Gomez Adan
'''

import Utils_Pln
import Reports
import spacy
from spacy.matcher import PhraseMatcher
from spacy import displacy
from pathlib import Path

# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('en_core_web_md')

oUtilsPln = Utils_Pln.UtilPln()
oReports = Reports.reports()

oUtilsPln.nlp = nlp
docBlank = nlp(' ')

tokenBlank = docBlank[0]

matcher = PhraseMatcher(nlp.vocab)

list_conjunctions_lower = ['provided that', 'as long as', 'so long as', 'in case',
                           'as soon as', 'on condition that', 'on the condition that',
                           'on the understanding that', 'only if', 'even if', 'as if']
list_conjunctions_upper = ['Provided that', 'As long as', 'So long as', 'In case',
                           'As soon as', 'On condition that', 'On the condition that',
                           'On the understanding that', 'Even if', 'If only', 'As if']

Conditional_patterns = []
Conditional_patterns.append([
    ['MV', ["VERB", "AUX"], ["ROOT", "MainVerb", "conj"]],
    ['VBC', ["VERB", "AUX"], ["advcl", "ccomp", "csubjpass", "xcomp", "auxpass", "conj"]],  #
    ['SW', ["ADP", "ADV", "VERB", "AUX", "SCONJ"], ["mark", "advmod", "prep", "csubj"]],
    ['1']
])
Conditional_patterns.append([
    ['MV', ["VERB", "AUX"], ["ROOT", "MainVerb", "conj"]],
    ['SW', ["ADP", "ADV", "VERB", "AUX", "SCONJ"],
     ["advmod", "prep", "ccomp", "acomp", "csubj", "advcl", "mark", "conj"]],
    # "pcomp" ,"mark"
    ['VBC', ["VERB", "AUX"], ["advcl", "ccomp", "acl", "csubj", "prep"]],  # pcomp",

    ['2']])

subOrdinanteAdverbialCondition = [
    'after',
    'as long as',
    'as soon as',
    'assuming',
    'before',
    'in case',
    'if',
    'unless',
    # 'lest',
    'providing',
    'provided that',
    'supposing',
    'so long as',
    'once',
    'only if',
    'on condition that',
    'on the condition that',
    'on the understanding that',
    'until',
    'when'
]

conjuntions_lower = [nlp.make_doc(text) for text in list_conjunctions_lower]
conjuntions_upper = [nlp.make_doc(text) for text in list_conjunctions_upper]

matcher.add('conjunctions_lower', None, *conjuntions_lower)
matcher.add('conjunctions_upper', None, *conjuntions_upper)


class Norm:

    def __init__(self):

        self.ID = ""
        self.TEXT = ""
        self.LANGUAGE = ""
        self.G_P_CLAUSE = ""
        self.G_Q_CLAUSE = ""
        self.G_SUBJECT = ""
        self.G_OBJECT = ""
        self.G_VERBS = ""
        self.G_MODALITY = ""
        self.G_PERSON1 = ""
        self.G_PERSON2 = ""
        self.G_P_VERB = ""
        self.G_P_VERB_TENSE = ""
        self.G_Q_VERB = ""
        self.G_Q_VERB_TENSE = ""
        self.G_STRUCTURE = ""
        self.G_ACTION = ""
        self.G_VOICE = ""
        self.G_P_VERB_TENSE_Q_VERB_TENSE = ""
        self.G_CANONICAL = ""

        self.a_subject = []
        self.a_object = []
        self.a_p_clause = []
        self.a_q_clause = []
        self.a_typeConditionals = []
        self.a_signalWordConditional = []
        self.a_tenseMainVerb = []
        self.a_tenseConditionalVerb = []
        self.a_modality = ""
        self.a_person1 = ""
        self.a_person2 = ""
        self.a_p_verb = ""
        self.a_p_verb_tense = ""
        self.a_q_verb = ""
        self.a_q_verb_tense = ""
        self.a_structure = ""
        self.a_action = ""
        self.a_verbs = ""
        self.a_voice = ""
        self.a_p_verb_tense_q_verb_tense = ""
        self.a_canonical = ""
        self.a_Pattern = ""
        self.a_SignalWord = ""

        self.preprocess = ""  # revisar
        self.MorphologicalPOST = ""
        self.MorphologicalNamedEntities = ""
        self.syntaxAnalysis_Dependency = ""
        self.analysis_01 = ""
        self.doc = ""

        self.evaluation_TAG_AUTOMATIC = []
        
        

    def process(self):

        self.doc = nlp(self.TEXT)
        for sentence in self.doc.sents:
          displacy.render(sentence, style='dep', jupyter=True, options={'distance': 90, 'collapse_phrases' : True, 'compact' : True, 'font':'Times New Roman'})
          svg = displacy.render(sentence, style="dep", jupyter= False, options={'distance': 90, 'collapse_phrases' : True, 'compact':True, 'font': 'Times New Roman'})
          output_path = Path("../results/images/"+self.ID[18:21]+".svg") # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script 
          output_path.open("w", encoding="utf-8").write(svg)
        #displacy.render(sentence, style='dep', jupyter=True, options={'distance': 80})

        # merge los token de subordinadas condicionales de mas de una palabra
        matches = matcher(self.doc)
        for match_id, start, end in matches:
            span = self.doc[start:end]
            #span.merge()

        with self.doc.retokenize() as retokenizer:
            for match_id, start, end in matches:
                span = self.doc[start:end]
                retokenizer.merge(span)

        self.linguistic_analysis()
        self.legal_analysis()

        return self

    def writeCSV(self, oNorms):

        oReports.write_file_results_Excel(oNorms)

    def displayTree(self, oNorm):

        self.displacy.serve(oNorm.doc, style='dep')

    def linguistic_analysis(self):  # for each norm

        svo, token_verb, token_subject, token_object, conditionals = self.SVOC(self.doc)

        predicates = [];
        objects = [];
        subjects = []

        PClause = [];
        QClause = [];
        signalWordFound = [];
        pattern = [];
        typeCondition = [];
        tenseQClause = []

        tensePClause = [];
        token_mainVerb = [];
        detail_Pattern = []

        for i in range(0, len(svo)):
            predicates.append(svo[i][1])
            subjects.append(svo[i][2])
            objects.append(svo[i][3])

        for c in conditionals:
            PClause.append(c[0])
            QClause.append(c[1])
            signalWordFound.append(c[2])
            pattern.append(c[3])
            typeCondition.append(c[4])
            tenseQClause.append(c[5])
            tensePClause.append(c[6])
            token_mainVerb.append(c[7])
            detail_Pattern.append(c[8])

        tokens_text = [];
        SignalWordConditional = "";
        CondLinkConRoot = "No";
        numberConditionals = 0
        PostWordConditional = "";

        for chunk in self.doc.noun_chunks:
            tokens_text.append([chunk.text])

        self.a_verbs = predicates
        self.a_subject = subjects
        self.a_object = objects

        self.a_p_clause = PClause
        self.a_q_clause = ""
        self.a_Pattern = pattern
        self.a_SignalWord = signalWordFound

        self.a_canonicalConditional = typeCondition

        self.token_verb = token_verb
        self.token_subject = token_subject
        self.token_object = token_object
        self.conditionals = conditionals
        self.signalWordConditional = signalWordFound
        self.tenseMainVerb = tenseQClause
        self.tenseConditionalVerb = tensePClause
        self.analysis_01 = detail_Pattern
        self.analysis_02 = CondLinkConRoot
        self.analysis_03 = SignalWordConditional
        self.analysis_04 = str(numberConditionals)
        self.analysis_05 = PostWordConditional
        self.analysis_06 = pattern
        self.analysis_07 = token_mainVerb
        self.analysis_08 = QClause

        self.analysis_16 = detail_Pattern  # ancestros

    def legal_analysis(self):
        verbs = self.a_verbs
        subjects = self.a_subject
        objects = self.a_object
        token_subjects = self.token_subject
        token_objects = self.token_object
        token_verbs = self.token_verb
        tokens_mainVerbs = [token[4] for token in token_verbs]

        conditionals = self.conditionals

        aEntities = []

        for ent in self.doc.ents:
            aEntities.append([ent.text, ent.label_])

        ahohfeld = []

        for i in range(0, len(verbs)):
            Hohfeld_Modality, Deontic_modality = self.found_modality(verbs[i])

            # if Hohfeld_Modality != "":
            p = self.found_modality_parameter(Hohfeld_Modality, aEntities, verbs[i], subjects[i], objects[i],
                                              token_subjects[i], token_verbs[i], token_objects[i], conditionals)

            ahohfeld.append([Hohfeld_Modality, p[0], p[1], p[2], p[3]])
            # for p in parameters_modality:
            #    ahohfeld.append(p)

        self.tag_legal_ahohfeld = ahohfeld

    def found_modality(self, predicate):
        # Modal System  (F.R. Palmer 2001)
        # Propositional modality : Epistemic or Evidence
        # Event modality: Deontic or Dynamic
        # can, may                                                   :Permission
        # must, ought, shall, will                                   :Obligation
        # cannot, may not, must not, ought not, shall not, will not  :Prohibition (Forbidden)

        P = ["can", "may", "could", "might"]
        O = ["must", "ought", "shall", "will", "would", "should", "'ll"]
        F = ["cannot", "can not", "can't", "may not", "must not", "mustn't", "ought not", "shall not", "will not",
             "won't"]  # ,"could not","should not","might not"]

        aux_verb = predicate[0]
        aux_neg = predicate[1]
        aux_verb_pass = predicate[2]
        verb = predicate[3]

        modal = (aux_verb + aux_neg).strip()

        Deontic_modality = "";
        AHohfeld_modality = ""

        # TODO Check When preposition of norm is a event (Deontic) or situation (epestemic)?
        # sense_epestimic =
        # sense_deontic   =

        if (modal in P):
            Deontic_modality = "Permission"
            AHohfeld_modality = "PRIVILEGE"
        elif (modal in O):
            Deontic_modality = "Obligation"
            AHohfeld_modality = "DUTY"
        elif (modal in F):
            Deontic_modality = "Prohibition"
            AHohfeld_modality = "NO_PRIVILEGE"

        # TODO Check Mapping Deontic Modality  TO A-Hohfeld Modalities (Duty)
        # TODO Check A-Hohfeld Modalities

        return Deontic_modality, Deontic_modality

    def SVOC(self, text_parser):

        mainVerbs = [];
        usedTag_ = [];
        values = [];
        texto = text_parser.text

        mainVerbs = self.found_MainVerbs(text_parser)

        results_text = [];
        Conditionals = []

        tokenVerb = [];
        tokenSubject = [];
        tokenObject = [];
        Conditions = [];
        TypeConditions = [];
        SignalWords = [];
        TenseMains = [];
        TenseConditionals = []

        for i in range(0, len(mainVerbs)):
            result, tokens = self.found_SVO(mainVerbs[i])

            mainVerbs[i][1] = result[0]  # [aux_verb, aux_neg, aux_verb_pass, verb]
            mainVerbs[i][2] = result[1]  # subject
            mainVerbs[i][3] = result[2]  # object

            tokenSubject.append(tokens[1])
            tokenObject.append(tokens[2])
            tokenVerb.append(tokens[0])

        tokenVerb, tokenSubject, tokenObject, mainVerbs = self.fix_SVO(mainVerbs, tokenVerb, tokenSubject, tokenObject)

        if len(mainVerbs) > 0:

            # Conditionals = self.find_Conditionals(mainVerbs, text_parser)
            Conditionals = self.find_Conditionals(tokenVerb, text_parser)

        else:
            Conditionals.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])

        values = [mainVerbs, tokenVerb, tokenSubject, tokenObject, Conditionals]

        return values

    def found_MainVerbs(self, parser_tokens):
        mainVerbs = [];
        mainVerb = None

        for token in parser_tokens:
            if token.pos_ in {"VERB", "AUX"} and token.dep_ == "ROOT":
                mainVerb = token
                mainVerbs = [token, '', '', '']

        if len(mainVerbs) > 0:
            mainVerbs = self.lookMoreMainVerbs(mainVerb, [mainVerbs])

        return mainVerbs

    def lookMoreMainVerbs(self, token_verb, mainVerbs):

        for token in token_verb.children:
            if token.dep_ in ('conj') and token.pos_ in ('VERB'):  # and token != token_verb:
                mainVerbs.append([token, '', '', ''])
                self.lookMoreMainVerbs(token, mainVerbs)

        return mainVerbs

    def found_modality_parameter(self, modality, aEntities, predicate, subject, object, token_subject, tokens_mainVerbs,
                                 token_object, conditionals):

        aux_verb = predicate[0]
        aux_neg = predicate[1]
        aux_verb_pass = predicate[2]
        token_mainVerb = predicate[3]

        # TODO Check PasiveVoice in norm text for
        isPasiveVoice = False

        Person1 = "";
        Person2 = "";
        action = "";
        cond = ""

        person1, person2 = self.found_modality_parameter_person1and2(modality, aEntities, token_subject,
                                                                     tokens_mainVerbs, token_object)
        # person2    = self.found_modality_parameter_person2(modality,aEntities, token_object,tokens_mainVerbs )
        cond = self.found_modality_parameter_conditionals(modality, conditionals, object, token_object)
        action = self.found_modality_parameter_action(modality, token_mainVerb, object, token_object, cond)

        return [person1, person2, action, cond]

    def found_SVO(self, mainVerbs):

        token_mainVerb = mainVerbs[0];
        text_mainVerb = token_mainVerb.text;

        subject = "";
        object = "";
        aux_verb = "";
        aux_verb_pass = "";
        aux_neg = "";
        objectComplement = "";
        tagUsed = [];

        tag = ""
        # tokens_verb=[[],[],[],[token_mainVerb.dep_+"-"+token_mainVerb.tag_,text_mainVerb]];

        tokens_subject = [];
        tokens_object = [];

        tokens_verb = [tokenBlank, tokenBlank, tokenBlank, tokenBlank,
                       token_mainVerb]  # 0:aux mod, 1:aux, 2:aux-neg, 3:auxpass, 4:verb

        tokens_compObject = []

        for token in token_mainVerb.children:
            if token.dep_ in ('neg'):
                tag = "neg"
                aux_neg += ''.join(w.text_with_ws for w in token.subtree)

                tokens_verb[2] = token

            elif token.dep_ in ("aux"):
                tag = "aux"

                aux_verb += ''.join(w.text_with_ws for w in token.subtree)
                if token.tag_ in ('MD'):
                    tokens_verb[0] = token  # [w.tag_, w.text]
                else:
                    tokens_verb[1] = token  # [w.tag_, w.text]

            elif token.dep_ in ("auxpass"):
                tag = "auxpass"
                aux_verb_pass += ''.join(w.text_with_ws for w in token.subtree)
                tokens_verb[3] = token  # = [w.tag_, w.text]
            elif token.dep_ in ("nsubj", "csubj", "csubjpass", "nsubjpass", "exp"):  # and subject=="": "agent",
                subject = ''.join(w.text_with_ws for w in token.subtree)
                tag = "subject"
                for w in token.subtree:
                    tokens_subject.append(w)  # [w.tag_, w.text,w.dep_, w.pos_]) #w.dep_+"-"+w.pos_+"-"+w.tag_

            elif token.dep_ in (
                    'pobj', 'dobj', 'ccomp', 'attr', 'xcomp', 'dative', 'iobj', 'oprd', 'acomp',
                    'agent'):  # 'acomp','conj'): # 'cc'):
                tag = "object"
                object += ''.join(w.text_with_ws for w in token.subtree)
                for w in token.subtree:
                    tokens_object.append(w)  # [w.tag_, w.text,  w.dep_, w.pos_]) #w.dep_+"-"+w.pos_+"-"+w.tag_


            elif token.dep_ in ('npadvmod', 'prep'):
                tag = "compObject"
                objectComplement += ''.join(w.text_with_ws for w in token.subtree)

                for w in token.subtree:
                    tokens_compObject.append(w)  # [w.tag_, w.text, w.dep_,w.pos_ ])

        if object != "" and objectComplement != "":
            object = object.strip() + " " + objectComplement.strip()
            for w in tokens_compObject:
                tokens_object.append(w)

        results = [[aux_verb, aux_neg, aux_verb_pass, text_mainVerb.strip()], subject.strip(), object.strip()]

        return results, [tokens_verb, tokens_subject, tokens_object]

    def fix_SVO(self, mainVerbs, token_verbs, token_subjects, token_objects):

        for i in range(0, len(mainVerbs)):
            auxverb = (mainVerbs[i][1][0] + mainVerbs[i][1][1] + mainVerbs[i][1][2]).strip()
            verb = mainVerbs[i][1][3]
            subject = mainVerbs[i][2]
            object = mainVerbs[i][3]

            if object == "":  # hacia adelante
                for j in range(i + 1, len(mainVerbs)):
                    if mainVerbs[j][3] != "" and mainVerbs[i][
                        3] == "":  # fix. para la busqueda con el primero que encuentra
                        mainVerbs[i][3] = mainVerbs[j][3]
                        token_objects[i] = token_objects[j]  # make change

            if subject == "":  # hacia atras
                for j in range(i - 1, -1, -1):
                    if mainVerbs[j][2] != "" and subject == "":
                        subject = mainVerbs[j][2]
                        mainVerbs[i][2] = subject
                        token_subjects[i] = token_subjects[j]  # make change
                # if subject == "" and mainVerbs[0][3] != "":
                #    subject = mainVerbs[0][3]
                #    mainVerbs[i][3] = subject

            if auxverb == "":  # hacia atras
                for j in range(i - 1, -1, -1):
                    if (mainVerbs[j][1][0] + mainVerbs[j][1][1] + mainVerbs[j][1][2]).strip() != "" and auxverb == "":
                        mainVerbs[i][1][0] = mainVerbs[j][1][0]
                        mainVerbs[i][1][1] = mainVerbs[j][1][1]
                        mainVerbs[i][1][2] = mainVerbs[j][1][2]
                        auxverb = (mainVerbs[i][1][0] + mainVerbs[i][1][1] + mainVerbs[i][1][2]).strip()

                        token_verbs[i][0] = token_verbs[j][0]
                        token_verbs[i][1] = token_verbs[j][1]
                        token_verbs[i][2] = token_verbs[j][2]
                        token_verbs[i][3] = token_verbs[j][3]

        # return mainVerbs, token_subjects, token_objects
        return token_verbs, token_subjects, token_objects, mainVerbs

    def find_Conditionals(self, tokens_mainVerbs, tokens_sentence):
        conditionals = []

        # SWs in sentence
        tokens_SW = []
        for token in tokens_sentence:
            if token.text.lower() in subOrdinanteAdverbialCondition:
                tokens_SW.append(token)

        # tokens_SW = [token for token in tokens_sentence if token.text.lower() in subOrdinanteAdverbialCondition]

        Pattern_text = ""

        token_mainVerbs = [token[4] for token in tokens_mainVerbs]

        for token_SW in tokens_SW:
            
            signalWord_included= False

                  
            for conditional in conditionals:
                print(conditional[9])
                if token_SW in conditional[9]:
                    signalWord_included = True

            if signalWord_included:
                continue

            tokensVBC = self.findVBC(token_SW, token_mainVerbs)

            existSyntacticPattern, path_pattern, token_MV, namePattern = self.sintacticPattern(token_SW, tokensVBC[4],
                                                                                               token_mainVerbs)  # token_MV[0])


            typeCanonicalPattern, tense_QClause, tense_PClause = self.canonicalPattern(tokens_mainVerbs, tokensVBC)

            txt_PClause_lower = [w.text.lower() for w in tokensVBC[4].subtree]
            txt_PClause = "".join([w.text_with_ws for w in tokensVBC[4].subtree])

            pClause_text, pClause_tokens = self.extract_PClause(token_SW, tokensVBC)

            #qClause = self.extract_QClause(token_SW, pClause_text, tokens_sentence)

            if existSyntacticPattern:
                conditionals.append([pClause_text,
                                     path_pattern,                                 
                                     token_SW.text,
                                     namePattern + " PVerb(" + tokensVBC[4].text + ") QVerb(" + token_MV.text + ")",
                                     typeCanonicalPattern,
                                     tense_QClause,
                                     tense_PClause,
                                     token_MV.text,
                                     Pattern_text, 
                                     pClause_tokens])

        # +"(" + token_MV[0].text + "-" +atoken_verb_PClause[0].text + ")"

        return conditionals

    def findVBC(self, token_sw, tokens_mainVerb):
        # 0:aux mod, 1:aux, 2:aux-neg, 3:auxpass, 4:verb

        tokensVBC = [tokenBlank, tokenBlank, tokenBlank, tokenBlank, tokenBlank]

        token_verb_up = tokenBlank
        token_verb_down = tokenBlank

        pos_sw = token_sw.i

        pos_verb_up = -1;
        pos_verb_down = -1
        for token in token_sw.children:  # token_sw.subtree:
            if token.pos_ in {"VERB", "AUX"} and token not in tokens_mainVerb:  # and token != token_sw:
                token_verb_down = token
                pos_verb_down = token.i
                break

        for token in token_sw.ancestors:
            if token.pos_ in {"VERB", "AUX"} and token not in tokens_mainVerb:
                token_verb_up = token
                pos_verb_up = token.i

                break

        if pos_verb_down > pos_sw:
            tokensVBC[4] = token_verb_down
        elif pos_verb_up > pos_sw:
            tokensVBC[4] = token_verb_up

        tokensVBC = self.foundAuxVerb(tokensVBC[4])

        return tokensVBC

    def foundAuxVerb(self, token_verb):
        # 0:aux mod, 1:aux, 2:aux-neg, 3:auxpass, 4:verb
        tokensVerb = [tokenBlank, tokenBlank, tokenBlank, tokenBlank, tokenBlank]
        for t in token_verb.children:
            if t.dep_ in ('aux'):
                if t.tag_ == "MD":
                    tokensVerb[0] = t

                else:
                    tokensVerb[1] = t
            elif t.dep_ in ('neg'):
                tokensVerb[2] = t

            elif t.dep_ in ('auxpass'):
                tokensVerb[3] = t

        tokensVerb[4] = token_verb

        return tokensVerb

    def sintacticPattern(self, SW, PVerb, tokens_QVerb):

        token_MV = tokens_QVerb[len(tokens_QVerb) - 1]  # self.tokenBlank

        # the 1 y 2 pattern

        for Qverb in tokens_QVerb:

            # verb_children = [token for token in Qverb.children]
            # Pverb_children = [token for token in token_PVerb.children]

            # if verb.pos_ in {"VERB", "AUX"} and \
            #         (token_PVerb in verb.children and token_PVerb.dep_ in {"advcl", "ccomp", "csubjpass", "xcomp", "auxpass", "conj"}) and \
            #         (token_SW in token_PVerb.children and token_SW.pos_ in {"ADP", "ADV", "VERB", "AUX", "SCONJ"}  and  token_SW.dep_ in {"mark", "advmod", "prep", "csubj"}):
            #     return True, '1', verb, '1'
            # if verb.pos_ in {"VERB", "AUX"} and \
            #         (token_SW in verb.children and token_SW.dep_ in {"advmod", "prep", "ccomp", "acomp", "csubj", "advcl", "mark", "conj"}) and \
            #         (token_PVerb in token_SW.children and token_PVerb.dep_ in {"advcl", "ccomp", "acl", "csubj", "prep"}):
            #     return True, '2', verb, '2'
            if (Qverb in PVerb.ancestors) and PVerb.dep_ in {"advcl", "ccomp", "csubjpass", "xcomp", "auxpass",
                                                             "conj"} and \
                    (PVerb in SW.ancestors and SW.dep_ in {"mark", "advmod", "prep", "csubj"} and SW.pos_ in {"ADP",
                                                                                                              "ADV",
                                                                                                              "VERB",
                                                                                                              "AUX",
                                                                                                              "SCONJ"}):
                return True, '1*', Qverb, '1*'
            if (Qverb in SW.ancestors) and SW.dep_ in {"advmod", "prep", "ccomp", "acomp", "csubj", "advcl", "mark",
                                                       "conj"} and \
                    (SW in PVerb.ancestors and PVerb.dep_ in {"advcl", "acl", "csubj", "prep", "ccomp"} and SW.pos_ in {
                        "ADP", "ADV", "VERB", "AUX", "SCONJ"}):
                return True, '2*', Qverb, '2*'

        return False, ' ', token_MV, 'X'

        # the 1* and 2* pattern

        # pattern = []
        #
        # lWeaklyConected = False
        # lStrongConected = False
        # for mv in tokens_QVerb:
        #     if token_SW in mv.children or token_PVerb in mv.children:
        #         token_MV = mv
        #         break
        #     elif mv in token_SW.ancestors or mv in token_PVerb.ancestors:
        #         token_MV = mv
        #         lWeaklyConected = True

        # if not lStrongConected:
        #     return False, pattern, token_MV, 'X'

        # Only Patterns *
        # for mv in tokens_MV:
        #    if  mv in token_SW.ancestors or mv in token_VBC.ancestors:
        #        token_MV = mv
        #        lWeaklyConected = True
        # if not lWeaklyConected:
        #     return False, pattern, token_MV, 'X'

        # pattern.append(['MV', token_MV.pos_, token_MV.dep_, token_MV.text])
        #
        # if token_PVerb in token_SW.children:
        #     pattern.append(['SW', token_SW.pos_, token_SW.dep_, token_SW.text])
        #     pattern.append(['VBC', token_PVerb.pos_, token_PVerb.dep_, token_PVerb.text])
        # elif token_SW in token_PVerb.children:
        #     pattern.append(['VBC', token_PVerb.pos_, token_PVerb.dep_, token_PVerb.text])
        #     pattern.append(['SW', token_SW.pos_, token_SW.dep_, token_SW.text])
        #
        #
        # if len(pattern) < 3:  # last pos is 'name pattern'
        #     return False, pattern, token_MV, 'X'
        #
        # for p in Conditional_patterns:
        #
        #     pattern_ok = True
        #
        #     for i in range(0, 3 ): # last pos is 'name pattern'
        #         if pattern[i][0] not in p[i][0] or pattern[i][1] not in p[i][1] or pattern[i][2] not in p[i][2]:
        #             pattern_ok = False
        #
        #     if pattern_ok:
        #         namePattern=  p[3] + ["*" if lWeaklyConected else ""]
        #         return True, pattern, token_MV, namePattern
        #
        # return False,pattern,token_MV,'X'

    def canonicalPattern(self, array_token_verbMainClause,
                         array_token_verbConditionalClause):  # tenseMainClause, tenseConditionalClause):

        token_verbMainClause = array_token_verbMainClause[0]
        token_verbConditionalClause = array_token_verbConditionalClause

        # if token_verbMainClause==token_verbConditionalClause or token_verbConditionalClause.text.strip()=="": # is the same verb
        #    return "","",""

        tenseMainClause = self.foundTenseVerb(token_verbMainClause)

        tenseConditionalClause = self.foundTenseVerb(token_verbConditionalClause)

        # Tipo 	    Uso	                                          Tiempo verbal 	                    Tiempo verbal
        # de oración                                               de la proposición                     de la proposición principal
        # condicional                                              "if"

        # Tipo 	    Uso	                                          Tiempo verbal 	                    Tiempo verbal
        # de oración                                               de la proposición                     de la proposición principal
        # condicional                                              "if"

        # Tipo 0	Hechos generales	                                Simple present	                    Simple present
        typeConditional = ""
        if tenseMainClause == "simplePresent" and tenseConditionalClause == "simplePresent":
            typeConditional = 0

        # Tipo 1	Una condición posible y su resultado probable       Simple present	                    Simple future
        elif tenseMainClause == "simpleFuture" and tenseConditionalClause == "simplePresent":
            typeConditional = 1

        # Tipo 2	Una condición hipotética y su resultado probable	Simple past	                        Present conditional o
        #                                                                                                   Present continuous conditional
        elif tenseMainClause in (
                "presentConditional", "presentContinuousConditional") and tenseConditionalClause == "pastSimple":
            typeConditional = 2

        # Tipo 3	Una condición no real del pasado y su resultado
        #           probable en el pasado	                            Past perfect	                    Perfect conditional
        elif tenseMainClause in (
                "perfectConditional", "perfectContinousConditional") and tenseConditionalClause == "pastPerfect":
            typeConditional = 3


        # Mixto	Una condición no real del pasado y su resultado
        #           probable en el presente	                            Past perfect	                    Present conditional
        elif tenseMainClause in ("presentConditional", "perfectConditional") and tenseConditionalClause in (
                "pastPerfect", "pastSimple"):
            typeConditional = 4

        return typeConditional, tenseMainClause, tenseConditionalClause

    def foundTenseVerb(self, array_tokenVerb):

        tokenVerb = array_tokenVerb[4]

        if tokenVerb == tokenBlank:
            return "withoutVerb"

        token_aux_modal = array_tokenVerb[0]
        token_aux = array_tokenVerb[1]
        token_auxpass = array_tokenVerb[3]

        auxVerb = ""
        for i in range(0, 1, 3):
            if array_tokenVerb[i] != tokenBlank:
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
        auxFuture = ["will", "shall", "will be"]  # ,"shall","can","could","may","might","should"]
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

        elif auxVerb in auxFuture and (tokenVerb.tag_ in verbBaseForm or token_auxpass.tag_ in verbBaseForm):
            tense = "simpleFuture"

        elif (
                tokenVerb.tag_ in verbSimplePresent or tokenVerb.tag_ in verbBaseForm or token_aux_modal.tag_ in verbSimplePresent or token_auxpass.tag_ in verbSimplePresent or token_aux.tag_ in verbSimplePresent) and auxVerb not in auxPast:
            tense = "simplePresent"

        elif tokenVerb.tag_ in verbPastSimple or token_auxpass.tag_ in verbPastSimple or auxVerb in auxPast:
            tense = "pastSimple"

        # El "past perfect" está compuesto por : el pasado del verbo to have (had) + el "past participle" del verbo principal.
        elif tokenVerb.tag_ in verbPastParticiple and auxVerb in auxPastPerfect:
            tense = "pastPerfect"

        if token_aux.pos_ in auxInfinitive:
            return "infinitive"

        return tense


    def extract_QClause(token_SW, pClause_text, tokens_sentence):

        pos_sentence_pClause = tokens_sentence.find( pClause_text)
        
        return ""

    def extract_PClause(self, token_signalWord,  tokensVBC):

        textConditional = "".join([w.text_with_ws for w in tokensVBC[4].subtree])

        
        pos_signalWord = textConditional.find(token_signalWord.text)

        if pos_signalWord < 0:
            txtClause = token_signalWord.text_with_ws + textConditional
        else:
            txtClause = textConditional[pos_signalWord:]

        doc_Clause = nlp(txtClause)

        existVerb = False
        existSubject = False
        pClause_txt = ""
        pClause_tokens= []

        if len(doc_Clause) < 3:
            return pClause_txt, pClause_tokens

      

        for i in range(0, len(doc_Clause)):
            token = doc_Clause[i]
            if token.text in (".", ",") and i == len(doc_Clause) - 1:
                continue

            nextToken = doc_Clause[i]

            if i + 1 < len(doc_Clause):
                nextToken = doc_Clause[i + 1]

            if token.text.lower() in (
                    ";", ",", "then") and existVerb and existSubject:  # and  nextToken.pos_ not in ("CCONJ") :
                break

            if token.pos_ in {"VERB", "AUX"}:
                existVerb = True
            if token.dep_ in {"nsubj", "csubj", "csubjpass", "nsubjpass", "exp"}:
                existSubject = True

            pClause_txt += token.text_with_ws
            pClause_tokens.append( token)

        return pClause_txt, pClause_tokens

    def found_modality_parameter_person1and2(self, modality, aEntities, tokens_subject, tokens_mainverbs,
                                             tokens_object):
        passiveVoice = False
        if tokens_mainverbs[3] != tokenBlank:
            passiveVoice = True

        tokens_person1 = [];
        tokens_person2 = [];
        bAgent = False

        if passiveVoice:
            for t in tokens_object:
                if t.dep_ in {"agent"} and t.head in tokens_mainverbs:
                    tokens_person1 = [t for t in t.children if t.pos_ in {"PROPN", "NNP", "PRON"}]
        else:
            for t in tokens_subject:
                if t.pos_ in {"PROPN", "PRON"} and t.head in tokens_mainverbs:
                    tokens_person1 = [t]
                    for t in t.children:
                        tokens_person1.append(t)

        for t in tokens_object:
            if t.dep_ in {"dative"} and t.head in tokens_mainverbs:
                tokens_person2 = [t for t in t.children if t.pos_ in {"PROPN", "NNP", "PRON"}]

        str_person1 = ''.join([t.text_with_ws for t in tokens_person1]) + ("*" if bAgent else "")
        str_person2 = ''.join([t.text_with_ws for t in tokens_person2]) + ("*" if bAgent else "")

        return str_person1, str_person2

    def found_modality_parameter_person2(self, modality, aEntities, token_object, tokens_mainverbs):
        person2 = ""
        for t in token_object:
            if t.dep_ in {"dative", "dobj"} and t.pos_ in {"PROPN", "NNP"} and t.head in tokens_mainverbs:
                person2 = t.text
                return person2
        return person2

    def found_modality_parameter_action(self, modality, verb, object, token_object, conditional):
        action = verb.strip() + "#" + object.strip()

        return action

    def found_modality_parameter_conditionals(self, modality, conditional, texto_O, token_object):

        condition = conditional

        return condition

    def MorphologicalAnalisys_EntitiesNamed_Norms(self):
        entities = []
        #for ent in oNorm.doc.ents:
        #    entities.append([ent.text, ent.start_char, ent.end_char, ent.label_])
        #return entities

        # tokens = nltk.word_tokenize(oNorm.texto)
        # tagged = nltk.pos_tag(tokens)
        # entities= nltk.chunk.ne_chunk(tagged, binary=True)
        # aEntities=[]
        # for e in self.oUtilsPln.leaves(entities, "NE"):
        #     aEntities.append(e)

        # TODO Entities Named compuests
        # grammar = "NP: {<NN.*><NN.*>}  # Chunk two consecutive nouns"
        # cp = nltk.RegexpParser(grammar)
        # p = cp.parse(tagged)
        #
        # for e in self.oUtilsPln.leaves(p, "NP"):
        #     aEntities.append(e)

    def syntaxAnalysis_Noun_chunks(sef):
        noun_chunks = []
        #for chunk in doc.noun_chunks:
        #    noun_chunks.append([chunk.text, chunk.root.text, chunk.root.dep_,
        #                        chunk.root.head.text])
        return noun_chunks

    def SyntaxAnalysis_Dependency_parsing(self):
        aDependency = []
        for token in self.doc:
            s = ""
            for c in token.children:
                s = s + str(c) + ";"

            aDependency.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                                token.shape_, token.is_alpha, token.is_stop, token.head.text, token.head.pos_, s])

        return aDependency

    def evaluation_tag_auotmatic(self):
        evaluation = []
        evaluation.append(["TAG A-hohfeld", "Manual", "Automatic", "Similarity"])
        manual_tag = self.oUtilsPln.foundTagsAhohfeld(self.TAG_AH_LEGAL_RELATION, True)
        for t in manual_tag:
            tag_manual = t[0].strip().upper()

            tag_automatic = self.oUtilsPln.found_value_tag(tag_manual, self.tag_ah_legal_relation)

            eval_tag = 100 if (tag_manual == tag_automatic) else 0

            evaluation.append([tag_manual, tag_manual, tag_automatic, eval_tag])

            parameters_manual = t[1]
            for i in range(0, len(parameters_manual)):
                x = parameters_manual[i].split("=")

                name_parameter_manual = x[0].strip().lower()
                value_parameter_manual = x[1].strip().lower()

                value_parameter_automatic = self.oUtilsPln.found_value_parameter(name_parameter_manual,
                                                                                 self.tag_ah_legal_relation)

                parm1 = self.nlp(value_parameter_manual)
                parm2 = self.nlp(value_parameter_automatic)
                eval_parm = round(parm1.similarity(parm2) * 100, 1)

                evaluation.append([name_parameter_manual, value_parameter_manual, value_parameter_automatic, eval_parm])

        return evaluation
