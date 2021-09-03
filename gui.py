from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
#pip install Pillow

# ? model libaries
import nltk
import re
from nltk.util import ngrams
nltk.download("punkt")
nltk.download('brown')
from nltk.corpus import brown
# from pathlib import Path
from nltk import edit_distance as ED
import heapq
from collections import Counter
import pandas as pd
# from pattern.en import conjugate, lemma, lexeme, PRESENT, SG
import itertools





# paths to files
aml_path = 'Corpus_PDFMiner_AMLtb.txt'
dl_path = 'Corpus_PDFMiner_DLtb.txt'
noisy_channel_path = "channel.xlsx"

"""# Text preprocessing
- exclude unwanted texts
- build unigram and bigram corpora

## Text cleaning
"""

# regular expression pattern

# website URL
website_pattern = r'(http|ftp|https+:\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?'

# email address
email_pattern = r'[\w\.-]+@[\w\.-]+'

# in-text citation
author = r"(?:[A-Z][A-Za-z'`-]+)"
etal = r"(?:et al\.?)"
additional = f"(?:,? (?:(?:and |& )?{author}|{etal}))"
year_num = "(?:19|20)[0-9][0-9]"
page_num = "(?:, p\.? [0-9]+)?"
year = fr"(?:, *{year_num}{page_num}| *\({year_num}{page_num}\))"
citation_pattern = fr'\b(?!(?:Although|Also)\b){author}{additional}*{year}'
citation_pattern_1 = fr'\b(?!(?:Although|Also|Similarly)\b){author}{additional}*{year}'
citation_pattern_2 = '\(([^()\d]*\d[^()]*)' # less formally written in-text citation

# abbreviation within parenthesis
abbrev_pattern = r'([A-Z]*\s?[A-Z]+[^a-z0-9\W])'

"""## Unigram corpus"""

# function for building unigram corpus without duplicate

def corpus(path, corpus_tokens):
	
	with open(path, 'r', encoding='utf8') as fileinput:
		for line in fileinput:
			line = line.lower()    #convert string into lowercases
			output = re.findall(email_pattern, line)    #capture email address in the string
			output2 = ["".join(x) for x in re.findall(website_pattern, line)]    #capture website in the string
			output3 = re.findall(citation_pattern, line) #capture formal in-text citation
			output4 = re.findall(citation_pattern_2, line) #capture improperly written in-text citation
			
			citation_output = list(set(output3 + output4)) #combine and remove duplication, if any
			
			words = nltk.word_tokenize(line)    #tokenize the string

			for word in words:
				if word.isalnum() and not word.isdigit() and len(word)>1 and word not in output and word not in output2 and word not in citation_output and word not in corpus_tokens:
					corpus_tokens.append(word)
	fileinput.close()   
	
	return corpus_tokens

# build unigram corpus with aml and dl textbooks without duplicate
unigram_corpus = corpus(aml_path, corpus_tokens=[])
unigram_corpus = corpus(dl_path, unigram_corpus)

# include brown corpus to augment unigram corpus
brown_words = set(brown.words())

brown_wordlist = []
for word in brown_words:
  brown_wordlist.append(word.lower())
len(brown_wordlist)

for word in brown_wordlist:
	if word not in unigram_corpus:
		unigram_corpus.append(word)

# print("number of unique words in unigram_corpus: ", len(unigram_corpus))

"""## Bigram corpus"""

# function for building unigram corpus with duplicates

def corpus2(path, corpus_tokens):
	
	with open(path, 'r', encoding='utf8') as fileinput:
		for line in fileinput:
			line = line.lower()    #convert string into lowercases
			output = re.findall(email_pattern, line)    #capture email address in the string
			output2 = ["".join(x) for x in re.findall(website_pattern, line)]    #capture website in the string
			output3 = re.findall(citation_pattern, line) #capture formal in-text citation
			output4 = re.findall(citation_pattern_2, line) #capture improperly written in-text citation
			
			citation_output = list(set(output3 + output4)) #combine and remove duplication, if any
			
			words = nltk.word_tokenize(line)    #tokenize the string

			for word in words:
				if word == 'a':
					corpus_tokens.append(word)
				elif word.isalnum() and not word.isdigit() and len(word)>1 and word not in output and word not in output2 and word not in citation_output:
					corpus_tokens.append(word)
	fileinput.close()   
	
	return corpus_tokens

# build unigram corpus with aml and dl textbooks with duplicates
corpus_tokens_dup = corpus2(aml_path, corpus_tokens=[])
corpus_tokens_dup = corpus2(dl_path, corpus_tokens_dup)

#number of unique words in our unigram corpus
V = len(list(set(corpus_tokens_dup)))

# build bigram corpus
bigram_corpus = list(ngrams(corpus_tokens_dup,2))
bigram_count = Counter(bigram_corpus)

# corpus_tokens_aml = corpus2(aml_path, corpus_tokens=[])
# corpus_tokens_dl = corpus2(dl_path, corpus_tokens=[])
# # print("number of words in AML textbook: ", len(corpus_tokens_aml))
# # print("number of words in DL textbook: ", len(corpus_tokens_dl))

# total_words = int(len(corpus_tokens_aml)) + int(len(corpus_tokens_dl))
# print("total number of words: ", total_words)

# uniq_tokens_aml = corpus(aml_path, corpus_tokens=[])
# uniq_tokens_dl = corpus(dl_path, corpus_tokens=[])
# print("number of unique words in AML textbook: ", len(uniq_tokens_aml))
# print("number of unique words in DL textbook: ", len(uniq_tokens_dl))

# print("total number of unique words: ", V)

"""# Non-word error

## Error detection
"""

def error_detection(input_text, corpus_tokens):

	error_list = []

	#perform regex operation on the entire input text, instead of on the word level
	citation_output = re.findall(citation_pattern_1, input_text)#must process before lower() since the pattern is sensitive to it
	citation_output = [x.lower() for x in citation_output]
	abrv_output = re.findall(abbrev_pattern, input_text) #regex based on capital so need to put before lower
	abrv_output = list(set([x.lower() for x in abrv_output])) #convert to lower for comparison 
	abrv_output = [j.replace(' ', '') for j in abrv_output] #remove space of the extracted abrv

	input_text = input_text.lower()

	email_output = re.findall(email_pattern, input_text)
	website_output = ["".join(x) for x in re.findall(website_pattern, input_text)]
	citation_output_2 = re.findall(citation_pattern_2, input_text)
	citation_output_2 = [x for x in citation_output_2 if len(x)!=4] #remove year in () from in-text citation
	
	mark_list = ['(', ')', '[', ']', ',', '.', ':', ';'] 

	#combine all regex output, except abbrv
	combined_output = list(set(email_output + website_output + citation_output + citation_output_2))

	updated_output = []

	#to remove in-text citation with last name of author only
	combined_citation = list(set(citation_output+citation_output_2))
	author_name = [k.split()[0] for k in combined_citation if ';' not in k]
	author_name_2 = [j.split()[0] for j in nltk.flatten([k.split(';') for k in combined_citation if ';' in k])]
	author_list = list(set(author_name+author_name_2))
	author_list

	#to make sure that the combined output is free from those marks and re-update to updated_output
	for item in combined_output:
		for mark in mark_list:
			if mark in item:
				item = item.replace(mark, '')
			else:
				continue
		updated_output.append(item)

	#to remove those regex output from the text, so that they're not involved during error detection
	#compare to both combined and updated to reduce the chance of miss captured words
	
	combined_output.sort(key=len, reverse=True) #prioritize longer string first, if not the short string will replace it
	updated_output.sort(key=len, reverse=True)

	#input_text = input_text.replace("’s", '') #different style of in-text citation
	apostrophe_result = re.findall("\w+\’s|\w+\'s", input_text)
	for i in apostrophe_result:
		if i in input_text:
			input_text = input_text.replace(i, '')

	for i in combined_output:
		if i in input_text:
			input_text = input_text.replace(i, '')

	for i in updated_output:
		if i in input_text:
			input_text = input_text.replace(i, '')

	for mark in mark_list:
		input_text = input_text.replace(mark, '')

	#check against the corpus
	input_words = input_text.split()
	for word in input_words:
		if len(word.lower())>1 and not word.isdigit() and word.lower() not in error_list and word.lower() not in abrv_output and word.lower() not in author_list:
			if word.lower() not in corpus_tokens:
				error_list.append(word)

	#remove word with dash which actually existed in corpus from error list            
	temp_list = []
	dash_mark = '-'
	for word in error_list:
		if dash_mark in word:
			split_word = word.split(dash_mark)
			for i in range(len(split_word)):
				if split_word[i] in corpus_tokens:
					temp_list.append(word)
					break
	 
	for j in temp_list:
		error_list.remove(j)
	
	return error_list

"""## Minimum edit distance

### Lavenshtein distance
"""

def word_suggestion(error_list, corpus, transOP=False, suggestion_no=5): #default to Levenshtein if not specify
	
	total_dict = {}
	
	for word1 in error_list:
		ED_dict = {}
		ED_dict1 = {}
		sort_dict = {}
		for word2 in corpus:
			if abs(len(word1) - len(word2)) <= 2:
				distance = ED(word1, word2, transpositions=transOP)
				ED_dict[word2] = distance
				if distance <= 1:
					ED_dict1[word2] = distance
		
		sort_dict['Top 5'] = heapq.nsmallest(suggestion_no, ED_dict, key=ED_dict.get)
		sort_dict['D1'] = heapq.nsmallest(suggestion_no, ED_dict1, key=ED_dict1.get)
		total_dict[word1] = sort_dict #use of nested dict to record error frequency if needed
		 
	return total_dict

"""## Noisy channel model"""

def noisy_channel_model(suggestions, path=noisy_channel_path):

	final_channel = {}
	error_list = [k for k in suggestions.keys()]
	potential_candidates = [v['D1'] for v in suggestions.values()]

	for i in range(len(error_list)):
		channel = {}
		error = error_list[i]
		candidates = potential_candidates[i]
		if len(candidates)<2: continue
		else:
			for word in candidates:

				if len(error) == len(word):

					dict1 = Counter(error)
					dict2 = Counter(word)

					if dict1 == dict2:
						#both words have same alphabets, transposition
						num = 0
						while num < len(error):
							#alphabets match
							if error[num] == word[num]:
								num += 1      
							else:
								edit = ''.join(reversed(word[num:num+2]))+'|'+word[num:num+2]
								edit = edit.replace(" ", "")
								break

					else:
						#substitution
						num = 0
						while num < len(error):
							#alphabets match
							if error[num] == word[num]:
								num += 1      
							else:
								edit = error[num]+'|'+word[num]
								edit = edit.replace(" ", "")
								break

				elif len(error) < len(word):
					#deletion
					num = 0
					while num < len(error):
						#alphabets match
						if error[num] == word[num]:
							num += 1      
						else:
							edit = word[num-1]+'|'+word[num-1:num+1]
							edit = edit.replace(" ", "")
							break

						#deletion at end of string
						if num == len(error):
							edit = word[num-1]+'|'+word[num-1:]
							edit = edit.replace(" ", "")

				else:
					#insertion
					num = 0
					while num < len(word):

						#alphabets match
						if error[num] == word[num]:
							num += 1      
						else:
							edit = '>'+error[num]+'|>'
							edit = edit.replace(" ", "")
							break

						#insertion at the end of string
						if num == len(word):
							edit = '>'+error[num]+'|>'
				channel[word] = edit
		final_channel[error] =channel
	
	#computation of probability
	df = pd.read_excel(path)
	final_dict = {}
	for k, v in final_channel.items():
		prob_dict = {}
		for k1, v1 in v.items():
			prob = df.loc[df['edit'] == v1, 'prob'].tolist()
			prob_dict[k1] = prob
		final_dict[k] = heapq.nlargest(len(v.items()), prob_dict, key=prob_dict.get)
	return final_dict

"""## Final suggestion"""

# Combine the output of word suggestion and noisy channel model
def combined_candidates(MED_cand, NC_cand):

	combined_cand = {}
	for k, v in MED_cand.items():
		if len(v['D1']) >= 2:
			combined_cand[k] = NC_cand[k]
			for candidate in v['Top 5']:
				if candidate not in NC_cand[k]:
					combined_cand[k].append(candidate)
		else:
			combined_cand[k] = v['Top 5']

	ranking_list = [1,2,3,4,5]

	for k, v in combined_cand.items():
		ranked_cand = dict(zip(ranking_list, v))
		combined_cand[k] = ranked_cand
		
	return combined_cand

def display_suggestion(suggestion_list):
	for k, v in suggestion_list.items():
		try:
			print(f"{k}: {v['Top 5']}\n")
		except KeyError:
			print(f"{k}: {v}\n")
		except TypeError:
			print(f"{k}: {v}\n")

"""## Combined function"""

def non_word(input_text, corpus_tokens):
	
	#generate the error list
	error_list = error_detection(input_text, corpus_tokens)
	
	#generate the suggestions using edit distance
	dsuggestions = word_suggestion(error_list, corpus_tokens, transOP=True)

	#generate the suggestions using noisy channel model
	NC = noisy_channel_model(dsuggestions)
	
	#combine suggestions from two functions
	txt_correction = combined_candidates(dsuggestions, NC)
	
	#print out the errors with their suggestions
	# nw_suggestion = display_suggestion(txt_correction)
	print(txt_correction)
	return txt_correction

"""# Real-word error

## Input text preprocessing
"""

# preprocessing for the input text for 
# -*- coding: utf-8 -*-

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences

"""## Generate candidate set"""

# mbleven algorithm

# Constants
REPLACE = 'r'
INSERT = 'i'
DELETE = 'd'
TRANSPOSE = 't'

MATRIX = [
	['id', 'di', 'rr'],
	['dr', 'rd'],
	['dd']
]

MATRIX_T = [
	['id', 'di', 'rr', 'tt', 'tr', 'rt'],
	['dr', 'rd', 'dt', 'td'],
	['dd']
]


# Library API

def compare(str1, str2, transpose=False):
	len1, len2 = len(str1), len(str2)

	if len1 < len2:
		len1, len2 = len2, len1
		str1, str2 = str2, str1

	if len1 - len2 > 2:
		return -1

	if transpose:
		models = MATRIX_T[len1-len2]
	else:
		models = MATRIX[len1-len2]

	res = 3
	for model in models:
		cost = check_model(str1, str2, len1, len2, model)
		if cost < res:
			res = cost

	if res == 3:
		res = -1

	return res


def check_model(str1, str2, len1, len2, model):
	"""Check if the model can transform str1 into str2"""

	idx1, idx2 = 0, 0
	cost, pad = 0, 0
	while (idx1 < len1) and (idx2 < len2):
		if str1[idx1] != str2[idx2 - pad]:
			cost += 1
			if 2 < cost:
				return cost

			option = model[cost-1]
			if option == DELETE:
				idx1 += 1
			elif option == INSERT:
				idx2 += 1
			elif option == REPLACE:
				idx1 += 1
				idx2 += 1
				pad = 0
			elif option == TRANSPOSE:
				if (idx2 + 1) < len2 and str1[idx1] == str2[idx2+1]:
					idx1 += 1
					idx2 += 1
					pad = 1
				else:
					return 3
		else:
			idx1 += 1
			idx2 += 1
			pad = 0

	return cost + (len1 - idx1) + (len2 - idx2)

#suggestions using mbleven algorithm

def word_suggestion2(word, corpus, suggestion_no=5): #default to Levenshtein if not specify
	
	suggestions = []
	ED_dict = {}
	for word2 in corpus:
		distance = compare(word, word2)
		
		#this is for mbleven
		if distance != -1:
			ED_dict[word2] = distance
			
	ED_dict = (dict(sorted(ED_dict.items(), key=lambda item:item[1])))
	ED_dict = dict(itertools.islice(ED_dict.items(), suggestion_no))
	suggestions = list(ED_dict.keys())
	
	return suggestions

"""## Noisy channel model"""

# suggestions using noisy channel model
def real_word(input_text):

	text = split_into_sentences(input_text)
	suggestions = {}

	for sentence in text:
		
		#tokenized the sentence and generate the bigrams
		words = nltk.word_tokenize(sentence)
		bigram_sentence = list(ngrams(words,2))
		count = 0
		
		for word in words[1:]: 
			
			if word.isalnum() and not word.isdigit():
				suggestion = ''

				#calcuate the probability of the original word
				first_bigram = bigram_sentence[count]
				second_bigram = bigram_sentence[count+1]
				
				first_bigram = tuple(item.lower() for item in first_bigram)
				second_bigram = tuple(item.lower() for item in second_bigram)
				
				max_prob = (bigram_count[first_bigram]+1)/len(bigram_corpus) * (bigram_count[second_bigram]+1)/len(bigram_corpus)

				#combine the variants of the word with the candidates
				# variants = lexeme(word.lower())
				candidates = word_suggestion2(word.lower(), corpus_tokens_dup)
				# candidates += variants
				candidates = list(set(candidates))

				#calculate the probablity of the other candidates
				for item in candidates:
					
					if item != word.lower():
						first_bigram1 = (first_bigram[0], item)
						second_bigram1 = (item, second_bigram[1])
						
						prob = (bigram_count[first_bigram1]+1)/len(bigram_corpus) * (bigram_count[second_bigram1]+1)/len(bigram_corpus)
 

						if prob>max_prob:
							max_prob = prob
							suggestion = item                         

				#add the suggestions into a dictionary
				if suggestion != '':
					suggestions[word] = [suggestion]
					
			count += 1

			#ending criteria for the for-loop
			if count == (len(bigram_sentence)-2):
					break
		
	ranking_list = [1]

	for k, v in suggestions.items():
		ranked_cand = dict(zip(ranking_list, v))
		suggestions[k] = ranked_cand

	return suggestions

"""## Capitalized error"""

def capitalized_error(input_text):
	
	capitalized_error = re.findall('(?<=[\.!?]\s)([a-z]+)', input_text)
	
	capitalized_suggestions = {}
	for error in capitalized_error:
		error_temp = '. '+ error.capitalize()
		error = '. ' + error
		capitalized_suggestions[error] = [error_temp]

	if len(re.findall('[a-z]', input_text[0])) !=0:
		first_error = re.findall('^([\w\-]+)', input_text)
		first_error = ''.join(first_error)
		capitalized_suggestions[first_error] = [first_error.capitalize()]
	
	ranking_list = [1]
	
	for k, v in capitalized_suggestions.items():
		ranked_cand = dict(zip(ranking_list, v))
		capitalized_suggestions[k] = ranked_cand
		
	return capitalized_suggestions

"""## Combined function"""

def real_words(input_text):
	
	#generate the suggestions for real-word errors
	suggestions = real_word(input_text)
	
	#generate the suggestions for lowercase letter errors
	capitalized_suggestions = capitalized_error(input_text)
	
	#combine suggestions from two functions
	suggestions.update(capitalized_suggestions)
	
	#print out the errors with their suggestions
	# rw_suggestion = display_suggestion(suggestions)
	
	return suggestions

	# combined_cand = {}
	# for k, v in MED_cand.items():
	# 	if len(v['D1']) >= 2:
	# 		combined_cand[k] = v['D1']
	# 		for candidate in v['Top 5']:
	# 			if candidate not in v['D1']:
	# 				combined_cand[k].append(candidate)
	# 	else:
	# 		combined_cand[k] = v['Top 5']
			
	# return combined_cand




class Nlp_gui:

	def __init__(self, master):
		# The header frame contains the name and logo
		master.resizable(False,False)
		self.frame_header = ttk.Frame(master)
		self.frame_header.pack()
		# This is the img part.
		self.logo = PhotoImage(file="nlp_logo.png").subsample(3,3)
		ttk.Label(self.frame_header, image=self.logo).grid(row = 0, column = 0,sticky = 'w')
		ttk.Label(self.frame_header, text="Best Natural Language Processing Group").grid(row =0, column =2, columnspan=3)

		# The content frame contains the "text input" and
		# "false words " and "sugesstion" and "output text" 
		self.frame_content = ttk.Frame(master)
		self.frame_content.pack()

		# ! this will print the user input as string.
		def getTextInput():
			global user_input_g
			user_input_g=self.input_text.get("1.0","end")
			testing()
			return user_input_g


		def selected_wordd(event):
			print(sugestion_treeview.selection())
			self.tuple_of_selection = sugestion_treeview.selection()
			

		def replace():
			list_user_input = self.input_text.get("1.0","end")
			list_corrected_words = []

			for string_dict in self.tuple_of_selection:
				temp_dict = eval(string_dict)
				list_corrected_words.append(temp_dict)

			for correction in list_corrected_words:
				for words in correction:
					list_user_input = list_user_input.replace(words,correction[words])

			self.input_text.replace("1.0","end", list_user_input)
			# print(list_user_input)

			print("------------start---------------")
			# print(list_corrected_words)
			# print(list_user_input)
			for i in sugestion_treeview.get_children():
				sugestion_treeview.delete(i)

			#? whatever that grammerly issue is goes here 
			real_user_input = self.input_text.get("1.0","end")
			
			issue_dict = real_words(real_user_input)
			# print(issue_dict)
			initial_index = 0
			child_index=0

			for issue in issue_dict:
				child_index=0
				sugestion_treeview.insert('',f"{initial_index}",issue, text=issue)
				for suggestion in issue_dict[issue]:
					sugestion_treeview.insert(issue,f"{suggestion}",{issue:issue_dict[issue][suggestion]}, text=issue_dict[issue][suggestion])
					child_index+=1
				sugestion_treeview.item(issue, open=True)
				initial_index+=1
			sugestion_treeview.bind('<<TreeviewSelect>>', selected_wordd)
			
			for highlighted_issue in issue_dict:
				started_character = real_user_input.find(highlighted_issue)
				print(started_character)
				end_character = started_character+len(highlighted_issue)
				self.input_text.tag_add('error',f"1.{started_character}" ,f"1.{end_character}")

			self.input_text.tag_config('error', background="lemon chiffon",foreground="red4")







		# ? This is the user input
		ttk.Label(self.frame_content, text = 'Text Input:').grid(row=1, column=0, padx=5, sticky='sw')
		self.input_text = Text(self.frame_content, width=50, height =30)
		self.input_text.grid(row=2, column=0, columnspan = 2,padx = 5, sticky = 'nw')
		# This is would be the list of the issued word.

		ttk.Label(self.frame_content, text = 'Spelling Error and Suggestions:').grid(row=1, column=2, padx=5, sticky='sw')
		# ? This is the Suggestion list.
		sugestion_treeview =ttk.Treeview(self.frame_content)
		sugestion_treeview.grid(row=2, column=2, padx=5, sticky='nw')

		# ? The footer will have two button "submit" and "clear"
		# the submit will run the algorithm and clear will gete all inputs. 
		self.frame_footer = ttk.Frame(master)
		self.frame_footer.pack()
		ttk.Button(self.frame_footer, text="Run", command=getTextInput).grid(row=3, column=0, columnspan = 2,padx = 5)
		ttk.Button(self.frame_footer, text="Replace", command=replace).grid(row=3, column=2, columnspan = 2,padx = 5)
		# ttk.Button(self.frame_footer, text="Clear").grid(row=3, column=4, columnspan = 2,padx = 5)

		# ? list of functions
		def testing():
			user_input_string = self.input_text.get("1.0","end")

			# error_list = error_detection(user_input_string)
			
			# total_dict = word_suggestion(error_list, corpus_tokens)
			# total_dict2 = noisy_channel_model(total_dict)
			# combined_candidates(total_dict,total_dict2)

			# issue_dict= combined_candidates(total_dict,total_dict2)
			issue_dict = non_word(user_input_string,unigram_corpus)
			# print(issue_dict)
			initial_index = 0
			child_index=0

			for issue in issue_dict:
				child_index=0
				sugestion_treeview.insert('',f"{initial_index}",issue, text=issue)
				for suggestion in issue_dict[issue]:
					sugestion_treeview.insert(issue,f"{suggestion}",{issue:issue_dict[issue][suggestion]}, text=issue_dict[issue][suggestion])
					child_index+=1
				sugestion_treeview.item(issue, open=True)
				initial_index+=1
			sugestion_treeview.bind('<<TreeviewSelect>>', selected_wordd)
			
			for highlighted_issue in issue_dict:
				started_character = user_input_string.find(highlighted_issue)
				print(started_character)
				end_character = started_character+len(highlighted_issue)
				self.input_text.tag_add('error',f"1.{started_character}" ,f"1.{end_character}")

			self.input_text.tag_config('error', background="lemon chiffon",foreground="red4")
			

			
			

			
		
		

# ! this the dictionary from out put
def testing_output():
	# dict_testing = {'loove':['love','live','light'], 'thi':['there','this','those','then']}
	dict_testing = {'loove':{1:'love',2:'live',3:'light'}, 'thi':{1:'there',2:'this',3:'these'}}
	return dict_testing


def main():
	# 1: Generate corpus
	

	root = Tk()
	feedback = Nlp_gui(root)
	root.mainloop()


if __name__ == "__main__": 
	# aml_path = 'Corpus_PDFMiner_AMLtb.txt'
	# dl_path = 'Corpus_PDFMiner_DLtb.txt'
	# #generate our own corpus
	# corpus_tokens = []
	# corpus_tokens = corpus(aml_path, dl_path, corpus_tokens)
	# #built-in default in nltk
	# wordlist = set(brown.words())
	# #final corpus
	# corpus_tokens += wordlist

	main()