from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image


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
		# ? This function will return string of userinputs.
		def getTextInput():
			global user_input_g
			user_input_g=self.input_text.get("1.0","end")
			return user_input_g

		# ? This is the user input
		ttk.Label(self.frame_content, text = 'User input:').grid(row =1, column = 0, padx = 5, sticky = 'sw')
		self.input_text = Text(self.frame_content, width=50, height =30)
		self.input_text.grid(row=2, column=0, columnspan = 2,padx = 5, sticky = 'nw')
		# This is would be the list of the issued word.

		ttk.Label(self.frame_content, text = 'Spelling Error:').grid(row =1, column = 2, padx = 5, sticky = 'sw')
		# This is the Suggestion list.
		ttk.Label(self.frame_content, text = 'Suggestion:').grid(row =1, column = 3, padx = 5, sticky = 'sw')
		# This is the output text box.
		ttk.Label(self.frame_content, text = 'Output:').grid(row =1, column = 4, padx = 5, sticky = 'sw')
		self.output_text = Text(self.frame_content, width=50, height =30)
		self.output_text.grid(row=2, column=4, columnspan = 2,padx = 5, sticky = 'nw')


        # The footer will have two button "submit" and "clear"
        # the submit will run the algorithm and clear will gete all inputs. 
		self.frame_footer = ttk.Frame(master)
		self.frame_footer.pack()
		ttk.Button(self.frame_footer, text="Run", command=getTextInput).grid(row=3, column=0, columnspan = 2,padx = 5)
		ttk.Button(self.frame_footer, text="Output").grid(row=3, column=2, columnspan = 2,padx = 5)
		ttk.Button(self.frame_footer, text="Clear").grid(row=3, column=4, columnspan = 2,padx = 5)

	




def main():
	root = Tk()
	feedback = Nlp_gui(root)
	root.mainloop()


if __name__ == "__main__": 
	main()