from reportlab.lib.pagesizes import letter
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import operator
# import svglib
from reportlab.graphics import renderPDF
import time
import Utils_Pln
import pandas as pd

import spacy

import csv


class reports():

    def __init__(self):
        self.oUtilsPln = Utils_Pln.UtilPln()

        self._PathFileLog = self.oUtilsPln.Read_Config('FilePath_LOG')[0] + str.replace(time.ctime(), ":", "-") + ".XML"
        self._PathFileImg = self.oUtilsPln.Read_Config('FilePath_IMG')[0]

    def write_file_results_xml(self, allNorms):
        self._FileLog = open(self._PathFileLog, "w", encoding="utf-8")
        self._FileLog.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        self._FileLog.write("<Results>\n")

        for oNorm in allNorms:

            self._FileLog.write("   <Norm>\n")
            self._FileLog.write("       <NormID>" + oNorm.ID + "</NormID>\n")
            self._FileLog.write("       <NormLanguage>" + oNorm.language + "</NormLanguage>\n")
            self._FileLog.write("       <NormText>" + str(oNorm.texto) + "</NormText>\n")
            self._FileLog.write("       <NormPreproces>" + str(oNorm.preprocess) + "</NormPreproces>\n")
            self._FileLog.write("       <NormMorphological_POST>\n")
            for m in oNorm.MorphologicalPOST:
                self._FileLog.write("               " + str(m) + '\n')
            self._FileLog.write("       </NormMorphological_POST>\n")

            self._FileLog.write("       <NormMorphological_NamedEntities>" + str(
                oNorm.MorphologicalNamedEntities) + "</NormMorphological_NamedEntities>\n")
            # self._FileLog.write("       <NormSyntaxAnalysis_Dependency>"+str(self.oNorm.syntaxAnalysis_Dependency)+"</NormSyntaxAnalysis_Dependency>\n")

            self._FileLog.write("       <NormSyntaxAnalysis_Dependency>\n")
            # self._FileLog.write("                  TEXT,	LEMMA,	POS,	TAG,	DEP,	SHAPE,	ALPHA,	STOP,	HEAD TEXT,	HEAD POS,	CHILDREN \n")
            for d in oNorm.syntaxAnalysis_Dependency:
                mark = "*" if (d[4] == "dobj" or d[4] == "pobj" or d[4] == "iobj" or d[4] == "nsubjpass") else " "
                self._FileLog.write("                 " + mark + str(d).strip('[]') + "\n")
            self._FileLog.write("       </NormSyntaxAnalysis_Dependency>\n")

            self._FileLog.write("       <NormSyntaxNoun_chunk>\n")
            # self._FileLog.write("                  TEXT,   ROOT.TEXT,  ROOT.DEP, ROOT.HEAD.TEXT \n")

            for n in oNorm.syntaxNoun_chunk:
                mark = "*" if (n[2] == "dobj" or n[2] == "pobj" or n[2] == "iobj" or n[2] == "nsubjpass") else " "
                self._FileLog.write("                 " + mark + str(n).strip("[]") + "\n")
            self._FileLog.write("       </NormSyntaxNoun_chunk>\n")

            self._FileLog.write("       <AHOHFELD_TAG_MANUAL>\n")

            ns = oNorm.ahohfeld_TAG_MANUAL.strip()
            ln = len(ns)
            f = ns.find(")")
            i = ns.find("(")
            d = 0
            while f < ln:
                nx = ns[i + 1:f].split(";")
                self._FileLog.write("                  " + ns[d:i] + "\n")
                for x in nx:
                    self._FileLog.write("                  " + x.strip() + "\n")
                d = f + 1
                i = d + ns[d:ln].find("(")
                f = d + ns[d:ln].find(")") if (ns[d:ln].find(")") > 0) else ln

            self._FileLog.write("       </AHOHFELD_TAG_MANUAL>\n")
            self._FileLog.write("       <AHOHFELD_TAG_AUTOMATIC>\n")
            for n in oNorm.ahohfeld_TAG_AUTOMATIC:
                self._FileLog.write("           " + str(n) + "\n")
            self._FileLog.write("       </AHOHFELD_TAG_AUTOMATIC>\n")

            self._FileLog.write("   </Norm>\n")

        self._FileLog.write("</Results>\n")
        self._FileLog.close()

        return True

    def write_file_results_pdf(self, allNorms):

        self.doc = SimpleDocTemplate("AnalysisNorms.pdf", pagesize=letter,
                                     rightMargin=12, leftMargin=12,
                                     topMargin=12, bottomMargin=12)
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        self.Story = []
        self.styleTitle = self.styles['Title']
        self.styleNormal = self.styles['Normal']
        self.styleHead1 = self.styles['Heading1']
        self.styleHead2 = self.styles['Heading2']
        self.styleBullet = self.styles['Bullet']
        self.styleDef = self.styles['Definition']
        self.styleItalic = self.styles['Italic']
        self.styleBodyText = self.styles['BodyText']
        self.styleCode = self.styles['Code']
        self.styleJustify = self.styles['Justify']

        self.totalesCat = {}  # key= CategoriaHohfeld + parameter + categoriaGramatical words

        self.totalesPam = {}  # key=  parameter + categoriaGramatical words

        self.Story.append(Paragraph("Analysis Norms", self.styleTitle))

        self.evaluation_general = [[]]

        for oNorm in allNorms:
            self.Story.append(Paragraph(
                '<font name=Courier size=8><u><b>01. NORM ID  :</b></u> %s</font>' % str(self.oNorm.id).zfill(4),
                self.styleJustify))
            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph('<font name=Courier size=8><u><b>02. TEXT:</b></u></font>', self.styleJustify))
            self.Story.append(
                Paragraph('<font name=Courier size=8>%s</font>' % self.oNorm.texto.strip(), self.styleNormal))
            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph(
                '<font name=Courier size=8><u><b>03. LANGUAGE:</b></u> %s</font>' % self.oNorm.language.strip().upper(),
                self.styleJustify))
            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph(
                '<font name=Courier size=8><u><b>04. COMMNUNITY:</b></u> %s</font>' % self.oNorm.community.strip(),
                self.styleJustify))
            self.Story.append(Paragraph('<font name=Courier size=8><b>URL:</b> %s</font>' % self.oNorm.url.strip(),
                                        self.styleJustify))
            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph('<font name=Courier size=8><u><b>05. MANUAL_TAG A-HOHFELD:</b></u></font>',
                                        self.styleJustify))
            self.Story.append(
                Paragraph('<font name=Courier size=8>%s</font>' % self.oNorm.ahohfeld_TAG_MANUAL, self.styleNormal))
            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph('<font name=Courier size=8><u><b>06. MEANING TAG A-HOHFELD:</b></u></font>',
                                        self.styleJustify))

            for t in self.oUtilsPln.foundTagsAhohfeld(self.oNorm.ahohfeld_TAG_MANUAL):
                self.Story.append(
                    Paragraph('<font name=Courier size=8>%s</font>' % self.oUtilsPln.explainTagAhohfeld(t),
                              self.styleNormal))

            self.Story.append(Spacer(0, 10))

            # if Onlylegal==False:

            self.Story.append(
                Paragraph('<font name=Courier size=8><u><b>07. PART OF SPEECH , ANALYSIS DEPENDENCY:</b></u></font>',
                          self.styleJustify))
            aMatrixDep = []
            aMatrixDep.append(['TEXT', 'LEMMA', 'POS', 'TAG', 'DEP', 'STOP', 'HEAD TEXT', 'HEAD POS', 'CHILDREN'])
            styleTable = [('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 6)]
            i = 0
            for d in oNorm.syntaxAnalysis_Dependency:
                namePOS = spacy.explain(d[2])
                nameTAG = spacy.explain(d[3])
                nameDEP = spacy.explain(d[4])
                aMatrixDep.append([d[0], d[1], namePOS, nameTAG, nameDEP, d[7], d[8], d[9], d[10]])
                i = i + 1
                if (d[2] == "VERB"):
                    styleTable.append(('BACKGROUND', (0, i), (10, i), colors.green))

                if (d[4] == "dobj" or d[4] == "pobj" or d[4] == "iobj" or d[4] == "nsubjpass"):
                    styleTable.append(('BACKGROUND', (0, i), (10, i), colors.yellow))

            t = Table(data=aMatrixDep, style=styleTable)
            self.Story.append(t)

            self.Story.append(
                Paragraph('<font name=Courier size=8><u><b>08. NOUN CHUNK:</b></u></font>', self.styleJustify))
            aNoun_chunk = []
            aNoun_chunk.append(['TEXT', 'ROOT.TEXT', 'ROOT.DEP', 'ROOT.HEAD.TEXT'])

            styleTable = [('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 6)]
            i = 0
            for d in oNorm.syntaxNoun_chunk:
                nameROOTDEP = spacy.explain(d[2])
                aNoun_chunk.append([d[0], d[1], nameROOTDEP, d[3]])
                i = i + 1
                if (d[2] == "VERB"):
                    styleTable.append(('BACKGROUND', (0, i), (3, i), colors.green))

                if (d[2] == "dobj" or d[2] == "pobj" or d[2] == "iobj" or d[2] == "nsubjpass"):
                    styleTable.append(('BACKGROUND', (0, i), (3, i), colors.yellow))

            t = Table(data=aNoun_chunk, style=styleTable)
            self.Story.append(t)

            drawing = svg2rlg(self._PathFileImg + str(self.oNorm.id).zfill(4) + ".svg")
            sx = sy = 0.25
            drawing.width, drawing.height = drawing.minWidth() * sx, drawing.height * sy

            drawing.scale(sx, sy)
            # if you want to see the box around the image
            # drawing._showBoundary = True

            t = Table([[drawing]], style=[('BACKGROUND', (0, 0), (0, 0), colors.black)])

            # im = Image(namefileimg, width=7 * inch, height=3 * inch)
            self.Story.append(Spacer(0, 10))
            self.Story.append(
                Paragraph('<font name=Courier size=8><u><b>09. GRAPHIC DEPENDENCY:</b></u></font>', self.styleJustify))

            # self.Story.append(drawing )
            self.Story.append(t)

            # self.Story.append(im)

            self.Story.append(Spacer(0, 10))
            self.Story.append(Paragraph('<font name=Courier size=8><u><b>10. ANALYSIS MANUAL TAG :</b></u></font>',
                                        self.styleJustify))

            ns = oNorm.ahohfeld_TAG_MANUAL.strip()
            ln = len(ns)
            # separar la etiquetas tag hohfeld
            f = ns.find(")")
            i = ns.find("(")
            d = 0

            while f < ln:
                # nombre etiqueta

                key = ns[d:i].strip()  # key= CategoriaHohfeld
                keyp = ""

                xnameTagHohfeld = '<font name=Courier size=8><b> %s </b></font>' % ns[d:i + 1]

                # recorre los parametros de la etiqueta
                for p in ns[i + 1:f].split(";"):
                    # nombre del parametro
                    p_i = p.find('=')

                    key = ns[d:i].strip() + ";" + p[0: p_i].strip()  # key= CategoriaHohfeld + parameter
                    keyp = p[0: p_i].strip()

                    statparam = p[p_i + 1:len(p)]
                    xnameTagHohfeld += "<br /><font name=Courier size=8><b> %s </b></font>" % p[0: p_i + 1]

                    # recorre palabras del parametro
                    for word in statparam.split():
                        bFound = False
                        xnameTagHohfeld += '<font name=Courier size=8> %s </font>' % word
                        for noun in self.oNorm.syntaxNoun_chunk:
                            if noun[0].strip().upper() == word.strip().upper():
                                key += ";" + noun[2].strip()  # CategoriaHohfeld + parameter + catg word
                                keyp += ";" + noun[2].strip()  # parameter + catg word
                                if (noun[2] == "VERB"):
                                    lab = '<font name=Courier size=8 backColor="green"> %s ' % "<" + spacy.explain(
                                        noun[2]).lower() + ">  </font>"
                                else:
                                    if (noun[2] == "dobj" or noun[2] == "pobj" or noun[2] == "iobj" or noun[
                                        2] == "nsubjpass"):
                                        lab = '<font name=Courier size=8 backColor="yellow"> %s ' % "<" + spacy.explain(
                                            noun[2]).lower() + ">  </font>"
                                    else:
                                        lab = '<font name=Courier size=8 backColor="white"> %s ' % "<" + spacy.explain(
                                            noun[2]).lower() + "> </font>"

                                xnameTagHohfeld += lab
                                bFound = True
                                break
                        if bFound == False:
                            for dep in self.oNorm.syntaxAnalysis_Dependency:
                                if dep[0].strip().upper() == word.strip().upper():
                                    key += ";" + dep[2].strip()  # CategoriaHohfeld + parameter + catg word
                                    keyp += ";" + dep[2].strip()  # CategoriaHohfeld + parameter + catg word

                                    if (dep[2] == "VERB"):
                                        lab = '<font name=Courier size=8 backColor="green"> %s ' % "<" + str(
                                            spacy.explain(dep[2])).lower() + "," + str(
                                            spacy.explain(dep[3])).lower() + "," + str(
                                            spacy.explain(dep[4])).lower() + '> </font>'
                                    else:
                                        if (dep[4] == "dobj" or dep[4] == "pobj" or dep[4] == "iobj" or dep[
                                            4] == "nsubjpass"):
                                            lab = '<font name=Courier size=8 backColor="yellow"> %s ' % "<" + str(
                                                spacy.explain(dep[2])).lower() + "," + str(
                                                spacy.explain(dep[3])).lower() + "," + str(
                                                spacy.explain(dep[4])).lower() + '> </font>'
                                        else:
                                            lab = '<font name=Courier size=8 backColor="white"> %s ' % "<" + str(
                                                spacy.explain(dep[2])).lower() + "," + str(
                                                spacy.explain(dep[3])).lower() + "," + str(
                                                spacy.explain(dep[4])).lower() + '> </font>'
                                    xnameTagHohfeld += lab
                                    break
                    self.oUtilsPln.count_Categoria(self.totalesCat, key)
                    self.oUtilsPln.count_Categoria(self.totalesPam, keyp)

                self.Story.append(Paragraph(xnameTagHohfeld, self.styleNormal))

                d = f + 1
                i = d + ns[d:ln].find("(")
                f = d + ns[d:ln].find(")") if (ns[d:ln].find(")") > 0) else ln

            self.Story.append(Spacer(0, 10))
            self.Story.append(
                Paragraph('<font name=Courier size=8><b><u>11. AUTOMATIC  TAG :</u></b></font>', self.styleJustify))

            self.Story.append(
                Paragraph('<font name=Courier size=7><b>11.1 LINGUISTIC TAG :</b></font>', self.styleJustify))

            self.Story.append(
                Paragraph('<font name=Courier size=8>%s</font>' % str(oNorm.linguistic_TAG1) + "<br />",
                          self.styleNormal))
            self.Story.append(
                Paragraph('<font name=Courier size=8>%s</font>' % str(oNorm.linguistic_TAG2) + "<br />",
                          self.styleNormal))

            self.Story.append(Spacer(0, 10))

            self.Story.append(Paragraph('<font name=Courier size=7><b>11.2 LEGAL TAG :</b></font>', self.styleJustify))

            for n in oNorm.ahohfeld_TAG_AUTOMATIC:
                self.Story.append(
                    Paragraph('<font name=Courier size=8>%s</font>' % str(n) + "<br />", self.styleNormal))

            self.Story.append(Spacer(0, 10))

            self.Story.append(Paragraph('<font name=Courier size=8><b><u>12. EVALUATION AUTOMATIC TAG :</u></b></font>',
                                        self.styleJustify))
            # self.Story.append(Paragraph('<font name=Courier size=8>%s</font>' % self.oNorm.ahohfeld_TAG_MANUAL, self.styleNormal))

            evaluation = []
            for x in oNorm.evaluation_TAG_AUTOMATIC:
                evaluation.append([x[0], Paragraph(x[1], self.styleJustify), Paragraph(x[2], self.styleJustify), x[3]])

            styleTable = [('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 6)]
            t = Table(data=evaluation, style=styleTable)
            self.Story.append(t)

            # acumula totales

            self.evaluation_general = self.eval_general(self.evaluation_general, oNorm.evaluation_TAG_AUTOMATIC)

            # if Onlylegal==True:
            #     self.Story.append(Spacer(0, 10))
            #
            # else:
            self.Story.append(PageBreak())

        # TODO Agregar totales del informe

        self.Story.append(PageBreak())

        self.Story.append(
            Paragraph('<font name=Courier size=8><b><u>13. GENERAL EVALUATION AUTOMATIC TAG :</u></b></font>',
                      self.styleJustify))
        # self.Story.append(Paragraph('<font name=Courier size=8>%s</font>' % self.oNorm.ahohfeld_TAG_MANUAL, self.styleNormal))

        self.Story.append(Spacer(0, 20))

        styleTable = [('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 6)]

        eval_general = []
        for x in self.evaluation_general:
            eval_general.append([Paragraph(x[0], self.styleNormal), Paragraph(str(x[3]), self.styleNormal)])

        t = Table(data=eval_general, style=styleTable)
        self.Story.append(t)

        # acumula totales

        self.Story.append(PageBreak())
        r = sorted(self.totalesPam.items(), key=operator.itemgetter(0))
        for i in r:
            self.Story.append(
                Paragraph('<font name=Courier size=8>%s ' % str(i[0]) + ":" + str(i[1]) + "</font>",
                          self.styleNormal))
        r = sorted(self.totalesCat.items(), key=operator.itemgetter(0))
        self.Story.append(PageBreak())
        for i in r:
            self.Story.append(
                Paragraph('<font name=Courier size=8>%s ' % str(i[0]) + ":" + str(i[1]) + "</font>",
                          self.styleNormal))

        self.doc.build(self.Story)

    def eval_general(self, tabla_general, tabla_parcial):
        if tabla_general == [[]]:
            tabla_general = tabla_parcial
            return tabla_general
        for x in range(0, len(tabla_parcial)):
            param = tabla_parcial[x][0]
            if (tabla_parcial[x][0] != "TAG A-hohfeld"):

                m = [i for i, x in enumerate(tabla_general) if x[0] == param]
                if m != []:
                    i = m[0]
                    tabla_general[i][3] += tabla_parcial[x][3]
                    tabla_general[i][3] = tabla_general[i][3] / 2
                else:
                    tabla_general.append(tabla_parcial[x])
        return tabla_general

    def write_file_results_Excel(self, allNorms):

        svo_ok = [0, 0, 0]
        svo_xx = [0, 0, 0]
        ah_ok = [0, 0, 0, 0, 0]
        ah_xx = [0, 0, 0, 0, 0]

        # ah_ok[4] = 0;
        # ah_xx[4] = 0
        norms_report = {}
        a_s = 0
        a_v = 0
        a_o = 0
        a_pClause = 0

        for oNorm in allNorms:
            text = oNorm.TEXT.strip()

            print("\n imprimiendo:", oNorm.ID)

            # check subject
            e_s = self.CompareWithGold(oNorm.G_SUBJECT, oNorm.a_subject)

            # check object
            e_o = self.CompareWithGold(oNorm.G_OBJECT, oNorm.a_object)

            # check verb
            e_v = self.CompareWithGold(oNorm.G_VERBS, oNorm.a_verbs)

            # check modality
            e_m = self.CompareWithGold(oNorm.G_MODALITY, oNorm.a_modality)

            # check person1

            e_p1 = self.CompareWithGold(oNorm.G_PERSON1, oNorm.a_person1)

            # check person2
            e_p2 = self.CompareWithGold(oNorm.G_PERSON2, oNorm.a_person2)

            # check action
            e_a = self.CompareWithGold(oNorm.G_ACTION, oNorm.a_action)

            # check conditionals

            e_qClause = self.CompareWithGold(oNorm.G_Q_CLAUSE, oNorm.a_q_clause)

            e_pVERB = self.CompareWithGold(oNorm.G_P_VERB, oNorm.a_p_verb)

            e_qVERB = self.CompareWithGold(oNorm.G_Q_VERB, oNorm.a_q_verb)

            e_qVERB_Tense = self.CompareWithGold(oNorm.G_Q_VERB_TENSE, oNorm.a_q_verb_tense)

            e_pVERB_Tense = self.CompareWithGold(oNorm.G_P_VERB_TENSE, oNorm.a_p_verb_tense)

            for index, condition in enumerate(oNorm.G_P_CLAUSE): #enumerate(oNorm.a_p_clause):
                e_pClause = self.CompareWithGold(oNorm.a_p_clause, [condition]) #e_pClause = self.CompareWithGold(oNorm.G_P_CLAUSE, [condition])
                norms_report[oNorm.ID + "-" + str(index)] = {'Id': oNorm.ID + "-" + str(index),
                                                             'Text': oNorm.TEXT,
                                                             'Subject': oNorm.a_subject,
                                                             "G_Subject": oNorm.G_SUBJECT,
                                                             "E_Subject": e_s,
                                                             "Verb": oNorm.a_verbs,
                                                             "G_Verb": oNorm.G_VERBS,
                                                             "E_Verb": e_v,
                                                             "Object": oNorm.a_object,
                                                             "G_Object": oNorm.G_OBJECT,
                                                             "E_Subject": e_s,
                                                             "Modality": oNorm.a_modality,
                                                             "G_Modality": oNorm.G_MODALITY,
                                                             "E_Modality": e_m,
                                                             "Person1": oNorm.a_person1,
                                                             "G_Person1": oNorm.G_PERSON1,
                                                             "E_Person1": e_p1,
                                                             "Person2": oNorm.a_person2,
                                                             "G_Person2": oNorm.G_PERSON2,
                                                             "E_Person2": e_p2,
                                                             "Conditional_Structure": oNorm.G_STRUCTURE,
                                                             "P-Clause": oNorm.a_p_clause,  # oNorm.a_p_clause,
                                                             "G_P-Clause": condition,  # oNorm.G_P_CLAUSE,
                                                             "E_P-Clause": e_pClause,
                                                             "Signal_Word": oNorm.a_SignalWord, #oNorm.a_SignalWord[index]
                                                             "G_Q-Clause": oNorm.G_Q_CLAUSE,
                                                             "Pattern": oNorm.a_Pattern, #oNorm.a_Pattern[index]
                                                             "G_P_VERB": oNorm.G_P_VERB,
                                                             "G_P_VERB_TENSE": oNorm.G_P_VERB_TENSE,
                                                             "G_Q_VERB": oNorm.G_Q_VERB,
                                                             "G_Q_VERB_TENSE": oNorm.G_Q_VERB_TENSE,
                                                             "G_VOICE": oNorm.G_VOICE,
                                                             "G_P_VERB_Tense_Q_VERB_TENSE": oNorm.G_P_VERB_Tense_Q_VERB_TENSE,
                                                             "G_CANONICAL": oNorm.G_CANONICAL
                                                             }

                a_s += e_s
                a_v += e_v
                a_o += e_o
                a_pClause += e_pClause

        df = pd.DataFrame.from_dict(norms_report, orient='index')
        df.to_excel('results.xlsx', sheet_name='results')

        print("\nSubject=", a_s, "\nVerb=", a_v, "\nObject", a_o, "\nPClause", a_pClause)

    def deprecated_CompareWithGold(self, gold, calculate):
        result = 0
        while "" in gold:
            gold.remove("")

        lenValue = len(calculate)
        lenGold = len(gold)

        if lenValue == 0 and lenGold == 0:
            return 1
        if (lenValue == 0 and lenGold != 0) or (lenGold == 0 and lenValue != 0):
            return 0

        for text in calculate:
            for ref in gold:
                if text.strip().lower() in ref.strip().lower() or ref.strip().lower() in text.strip().lower():
                    result += 1

        return result / lenGold

    def CompareWithGold(self, gold, calculate):
        result = 0
        while "" in gold:
            gold.remove("")


        lenCalculate = len(calculate)
        lenGold = len(gold)

        if lenCalculate == 0 and lenGold == 0:
            return 1
        if (lenCalculate == 0 and lenGold != 0) or (lenGold == 0 and lenCalculate != 0):
            return 0

        for text in calculate:
            for ref in gold:
                if text.strip().lower() in ref.strip().lower():
                    return 1

        return result
