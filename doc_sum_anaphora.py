#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from nltk.tag import StanfordNERTagger


st = StanfordNERTagger('/Users/ACER/Desktop/gitExp/anaphora/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/Users/ACER/Desktop/gitExp/anaphora/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner.jar',
					   encoding='utf-8')
def create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

text_string='''Vincent Willem van Gogh was a Dutch post-impressionist painter who posthumously became one of the most famous and influential figures in the history of Western art. In a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life. They include landscapes, still lifes, portraits and self-portraits, and are characterised by bold colours and dramatic, impulsive and expressive brushwork that contributed to the foundations of modern art. He was not commercially successful, and his suicide at 37 came after years of mental illness, depression and poverty. Born into an upper-middle-class family, Van Gogh drew as a child and was serious, quiet, and thoughtful. As a young man he worked as an art dealer, often travelling, but became depressed after he was transferred to London. He turned to religion and spent time as a Protestant missionary in southern Belgium. He drifted in ill health and solitude before taking up painting in 1881, having moved back home with his parents. His younger brother Theo supported him financially, and the two kept a long correspondence by letter. His early works, mostly still lifes and depictions of peasant labourers, contain few signs of the vivid colour that distinguished his later work. In 1886, he moved to Paris, where he met members of the avant-garde, including Emile Bernard and Paul Gauguin, who were reacting against the Impressionist sensibility. As his work developed he created a new approach to still lifes and local landscapes. His paintings grew brighter in colour as he developed a style that became fully realised during his stay in Arles in the south of France in 1888. During this period he broadened his subject matter to include series of olive trees, wheat fields and sunflowers. Van Gogh suffered from psychotic episodes and delusions and though he worried about his mental stability, he often neglected his physical health, did not eat properly and drank heavily. His friendship with Gauguin ended after a confrontation with a razor when, in a rage, he severed part of his own left ear. He spent time in psychiatric hospitals, including a period at Saint-Remy. After he discharged himself and moved to the Auberge Ravoux in Auvers-sur-Oise near Paris, he came under the care of the homeopathic doctor Paul Gachet. His depression continued and on 27 July 1890, Van Gogh shot himself in the chest with a Lefaucheux revolver. He died from his injuries two days later.'''

        
sentences=sent_tokenize(text_string)


freqTable=create_frequency_table(text_string)
print(freqTable)
def score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue
sentenceValue= score_sentences(sentences, freqTable)
print(sentenceValue)
def find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = int(sumValues / len(sentenceValue))

    return average
def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary
threshold= find_average_score(sentenceValue)
print(threshold)
print(generate_summary(sentences, sentenceValue, threshold))


#print(sentences)
tokenized_text = word_tokenize(text_string)

classified_text = st.tag(tokenized_text)


new_text_string=''
anaphora_p=['he','his','she','her','His','Her','He','She']

pointer_person=''
person=''

for i in range(0,len(tokenized_text)):
    a=tokenized_text[i] 
    b=classified_text[i][1]
    if b=='PERSON':
        pointer_person=a
 
    else:
        if a in anaphora_p:
            if len(person)==0:
                a=pointer_person
            else:
                a=person
        if a=='.' or a=='!' or a=='?':
            person=pointer_person
            #print(person)
        new_text_string+=a+' '
    
#print(new_text_string) 
sentences=sent_tokenize(new_text_string)
freqTable=create_frequency_table(new_text_string)
sentenceValue= score_sentences(sentences, freqTable)
threshold= find_average_score(sentenceValue)
print(freqTable, sentenceValue, threshold)
print('\n')
print(generate_summary(sentences, sentenceValue, threshold))
