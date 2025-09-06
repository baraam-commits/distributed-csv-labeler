import time
import ollama
from Llm_classifer_script import llmClassifier as classifier

def _save_question_map(questions, classifications, path="question_map.json"):
    from collections import OrderedDict
    import json
    if len(questions) != len(classifications):
        raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(classifications)} classifications")

    # Preserve insertion order (regular dicts do in Python 3.7+, OrderedDict for clarity)
    mapping = OrderedDict()
    for q, cls in zip(questions, classifications):
        if not isinstance(cls, dict):
            raise TypeError(f"Classification for '{q[:60]}...' is not a dict: {type(cls)}")
        mapping[q] = cls

    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


questions = [
            "who defines the specifications of a product or service",
            "who made the world's first computer virus",
            "what are the ingredients in herb de provence",
            "where is the world's largest swimming pool located",
            "age of the planet of the apes full movie",
            "who were involved in the battle of gettysburg",
            "what is the average climate of the taiga",
            "when does season 5 of senora acero come out",
            "when does the casino open in springfield ma",
            "who said no to this is your life",
            "show me a map of the arctic circle",
            "what was the rationale behind the american invasion of canada",
            "who is the killer in and then there were fewer",
            "btm 1st stage comes under which police station",
            "who is the chairman of google at present",
            "who have been the hosts of the price is right",
            "who sang the song mama told me not to come",
            "who plays grey worm on game of thrones",
            "working principle of high pressure sodium vapour lamp",
            "what is the latest red hat linux version",
            "What are the treatments for Lymphocytic Choriomeningitis (LCM) ?",
            "how can botulism be prevented?",
            "how can y. enterocolitica infections be treated for Yersinia ?",
            "what are public health agencies doing to prevent or control yersiniosis for Yersinia ?",
            "What is (are) Parasites - Lymphatic Filariasis ?",
            "Who is at risk for Parasites - Lymphatic Filariasis? ?",
            "How to diagnose Parasites - Lymphatic Filariasis ?",
            "What are the treatments for Parasites - Lymphatic Filariasis ?",
            "How to prevent Parasites - Lymphatic Filariasis ?",
            "What is (are) Parasites - Loiasis ?",
            "Who is at risk for Parasites - Loiasis? ?",
            "How to diagnose Parasites - Loiasis ?",
            "What are the treatments for Parasites - Loiasis ?",
            "I searched for the cause of this error and found that I have to change permissions or run gulp using sudo, but still got the same error. Can anyone please help... internal/child_process.js:298 throw errnoException(err, 'spawn'); ^ Error: spawn EACCES at exports._errnoException (util.js:870:11) at ChildProcess.spawn (internal/child_process.js:298:11) at exports.spawn (child_process.js:362:9) at exports.execFile (child_process.js:151:15) at ExecBuffer. (/var/www/conFusion/node_modules/gulp-imagemin/node_modules/imagemin/node_modules/imagemin-optipng/node_modules/exec-buffer/index.js:91:3) at /var/www/conFusion/node_modules/gulp-rev/node_modules/vinyl-file/node_modules/graceful-fs/graceful-fs.js:42:10 at /var/www/conFusion/node_modules/gulp-cache/node_modules/cache-swap/node_modules/graceful-fs/graceful-fs.js:42:10 at /var/www/conFusion/node_modules/gulp-imagemin/node_modules/imagemin/node_modules/vinyl-fs/node_modules/graceful-fs/graceful-fs.js:42:10 at FSReqWrap.oncomplete (fs.js:82:15)",
            "here is my code for select details by checking the name given. here I want to do name filter by starting letter also.. how can I do it? select * from tblcustomer where customername=case @customername when null then customername else @customername",
            "I'm having trouble understanding and using Django's ImageField. I have a model: My file system is currently: When I run the server and go to the Admin page, I can add BlogContent objects. After choosing an image for the image field, the image has a temporary name. However, after I save this object I can't find the image in the folder specified by the upload_to path. What is the correct way to do this?",
            "I'm planning to use SASS in making a website instead of CSS. I'm trying to compile a SASS file to CSS. Should I use an application to do so? Or should I use just command prompt?",
            "I am new here, i need to ask that, how can i get the value of a textbox and store it outside the FORM.cs, and to get that value to show it on a label... it is just for test application.. i want to code that is independent from GUI. thanks for your help in advance. here is what i was trying... In Form.cs private void button4_Click(object sender, EventArgs e) { cueTextBox2.Text = value; Calling cal = new Calling(); cal.setntags(value); } in Calling.cs public string setntags(string value) { value = tag1; MessageBox.Show(\"done\"); return tag1; } i am new in coding.. please help me,.",
            "class Cylinder(object): self.pi = 3.14 def __init__(self,height=1,radius=1): self.height = height self.radius = radius def volume(self): return self.pi * self.radius**2 * self.height def surface_area(self): pass",
            "I need a little help. I'm try to UPDATE a datetime to MySQL, but it didn't work. The declaration is like this: After this i want to UPDATE, but in MySQL is still blank always. UPDATE: If i use NOW() instead of \".$time.\", it works perfectly. If someone can help, please write the solution. Thanks, KoLi",
            "To give a little context to my issue... I have a Java EE web application (as a UI / client) that accesses services for data / business logic via a REST interface using the JAX-RS 2.0 client API (Resteasy implementation). Currently I inject a new JAXRS Client instance per request using a RequestScoped CDI managed bean, the thinking being that the client app may call multiple backend resources per request and I reuse the same JAXRS Client for the whole request (although I read somewhere this may not be correct as I am potentially changing the URI for each invocation) The documentation for JAXRS Client seems to suggest that the client is a potentially expensive operation and the app should limit the amount of connections it creates. It also seems to contradict itself and suggest the client should be closed once all the requests to a particular WebTarget are finished. The client application could potentially support thousands of simultaneous users so creating and destroying thousands of 'expensive clients' does not seem to be the correct approach so am thinking a shared client pool is more appropriate but there doesn't seem to be any information on how this should be achieved. All examples appear to show creating a new client for the request and a) closing it after or b) not closing it but not really explaining what happens on a second request. Can you help provide some answers on how you think this would be solved or information on what the best practice for this approach is. Thanks.",
            "I want to build a list of words. For each word on each line check to see if the word is already in the list and if not append it to the list. When the program completes, sort and print the resulting words in alphabetical order. But when I add sting to list ,it shows \"argument of type'NoneType' is not itrable\".What' worry? fh = (\"But soft what light through yonder window breaks\" \"It is the east and Juliet is the sun\" \"Arise fair sun and kill the envious moon\" \"Who is already sick and pale with grief\") lst = list() for line in fh: words = line.split() for word in line: if word not in lst: lst = lst.append(word) lst.sort() print lst",
            "//Ticket parent class import java.util.ArrayList; import java.util.Scanner; import java.io.*; public class Ticket { public Ticket() { seatArray = new ArrayList (); } public void loadIn() { //generic seating for plays and concerts seatArray.add(new TicketObject(\"A1\", 40)); seatArray.add(new TicketObject(\"A2\", 40)); seatArray.add(new TicketObject(\"A3\", 40)); seatArray.add(new TicketObject(\"A4\", 40)); seatArray.add(new TicketObject(\"A5\", 40)); seatArray.add(new TicketObject(\"B1\", 35)); seatArray.add(new TicketObject(\"B2\", 35)); seatArray.add(new TicketObject(\"B3\", 35)); seatArray.add(new TicketObject(\"B4\", 35)); seatArray.add(new TicketObject(\"B5\", 35)); } public String getSeats(int x) { return seatArray.get(x).getName() + \" $\" + seatArray.get(x).getPrice(); } protected ArrayList seatArray; } //Concert ticket child class import java.util.ArrayList; import java.util.Scanner; import java.io.*; public class ConcertTicket extends Ticket { public ConcertTicket() { super(); } public void loadIn() { super.loadIn(); //Special option for concerts seatArray.add(new TicketObject(\"Backstage Pass\", 100)); seatArray.add(new TicketObject(\"Backstage Pass\", 100)); seatArray.add(new TicketObject(\"Backstage Pass\", 100)); seatArray.add(new TicketObject(\"Backstage Pass\", 100)); seatArray.add(new TicketObject(\"Backstage Pass\", 100)); } } //Play ticket child class import java.util.ArrayList; import java.util.Scanner; import java.io.*; public class PlayTicket extends Ticket { public PlayTicket() { super(); } public void loadIn() { super.loadIn(); //Specialized seating for plays seatArray.add(new TicketObject(\"Box 1\", 150)); seatArray.add(new TicketObject(\"Box 2\", 150)); seatArray.add(new TicketObject(\"Box 3\", 150)); seatArray.add(new TicketObject(\"Box 4\", 150)); seatArray.add(new TicketObject(\"Box 5\", 150)); } } //Ticket object for each seat; also used to populate array public class TicketObject { public TicketObject(String inSeatName, int inSeatPrice) { seatName = inSeatName; seatPrice = inSeatPrice; } public String getName() { return seatName; } public int getPrice() { return seatPrice; } private String seatName; private int seatPrice; } //Price calculation class public class Calculations { public static double addTax(int total) { return total * 1.07; } } //Tester class import java.util.Scanner; public class TicketTester { public static void main(String[] args) { Scanner in = new Scanner(System.in); try { //Introduction for user System.out.println(\"Welcome to the ticket purchasing program!\"); System.out.println(\"You will be able to purchase either a concert or play ticket\"); System.out.println(\"\\n\"); System.out.println(\"Type concert or play to display available seats and deals\"); running = true; //Displaying tickets of either a concert or play String answer = in.next(); if (answer.equalsIgnoreCase(\"Concert\")) { ConcertTicket journeyConcert = new ConcertTicket(); journeyConcert.loadIn(); System.out.println(\"Seat Price\"); for(int i=0;i<15;i++) { System.out.println(journeyConcert.getSeats(i)); } System.out.println(\"\\n\"); System.out.println(\"Type in a seat name/option and press enter to reserve it.\"); System.out.println(\"Type purchase and press enter to finalize your ticket purchase\"); } else if(answer.equalsIgnoreCase(\"Play\")) { PlayTicket catsPlay = new PlayTicket(); catsPlay.loadIn(); System.out.println(\"Seat Price\"); for(int i=0;i<15;i++) { System.out.println(catsPlay.getSeats(i)); } System.out.println(\"\\n\"); System.out.println(\"Type in a seat name/option and press enter to reserve it.\"); System.out.println(\"Type purchase and press enter to finalize your ticket purchase\"); } else System.out.println(\"Enter a valid input\"); //Adding up chosen seat numbers and costs while(running) { String seatChoice = in.next(); if ((seatChoice.substring(0,1)).equalsIgnoreCase(\"A\") && seatChoice.length() == 2) { total = total + 40; seats = seats + 1; } else if ((seatChoice.substring(0,1)).equalsIgnoreCase(\"B\")&& seatChoice.length() == 2) { total = total + 35; seats = seats + 1; } else if ((seatChoice.substring(0,3)).equalsIgnoreCase(\"Box\")&& seatChoice.length() == 5) { total = total + 150; seats = seats + 1; } else if ((seatChoice.substring(0,14)).equalsIgnoreCase(\"Backstage Pass\") && seatChoice.length() == 14) { total = total + 100; seats = seats + 1; } else if (seatChoice.equalsIgnoreCase(\"Purchase\")) { System.out.println(\"You reserved \" + seats + \" seats at a price of $\" + Calculations.addTax(total)); running = false; } else System.out.println(\"Enter a valid input\"); } } finally { if(in!=null) in.close(); } } private static Boolean running; private static int total = 0; private static int seats = 0; } When I run the TicketTester class, everything runs correctly until I attempt to add either a backstage pass or a box office seat. I am given this error: \"Exception in thread \"main\" java.lang.StringIndexOutOfBoundsException: String index out of range: 14 at java.lang.String.substring(Unknown Source) at TicketTester.main(TicketTester.java:67)\" From this I can see that the error is occurring at the following line, but I don't know how to resolve it. else if ((seatChoice.substring(0,13)).equalsIgnoreCase(\"Backstage Pass\") && seatChoice.length() == 14)",
            "In Python-telegram-bot how to get, if possible , the complete list of all participants of the group at which the bot was added?",
            "Let's say you have the repository: Over time (months), you re-organise the project. Refactoring the code to make the modules independent. Files in the megaProject directory get moved into their own directories. Emphasis on move - the history of these files is preserved. Now you wish to move these modules to their own GIT repos. Leaving the original with just megaProject on its own. The filter-branch command is documentated to do this but it doesn't follow history when files were moved outside of the target directory. So the history begins when the files were moved into their new directory, not the history the files had then they lived in the old megaProject directory. How to split a GIT history based on a target directory, and, follow history outside of this path - leaving only commit history related to these files and nothing else? The numerous other answers on SO focus on generally splitting apart the repo - but make no mention of splitting apart and following the move history.",
            "I´ve came across the following error. At the moment I developing an Android App with React Native therefore I´m planning to use fetch for doing a post request for me. The app now throws an error: When I change the code to a GET-Request it´s working fine, in the browser with a window.alert() as a return it´s cool and also the chrome extension Postman returns data correctly.",
            "(f(n)) and O(f(n)) Can someone please give the mathematical definition of (f(n)) and O(f(n))?",
            "My exploration here comes from a recent Arduino project. I have an old(ish) Android LG Tribute. I removed the broken screen so now the device is missing visual output. I have rooted it and can control it from shell commands and other ways. I want to write an application for the device to communicate over usb. Basically what I want to accomplish: Plug in USB cord to android. Press button on cord plugged into android device -> Snap picture with devices camera - Output data in visual blinks via LED programmed blink logic for debug. I feel I should be able to handle nearly everything on the device. I just need to know where to begin looking for Android USB output and input programming. Basically want to use my android as a microcontroller...",
            "Like instance, in online purchasing a bill is created. I want to insert the items into the array and display it along with the rates. But I am unable to insert the items into the array. How do I do that?",
            "I'm using the Elasticsearch Bulk API to create or update documents. I do actually know if they are creates or updates, but I can simplify my code by just making them all index , or \"upserts\" in the SQL sense. Is there any disadvantage in using index (and letting ES figure it out) over using the more explicit create and update ?",
            "I have a website (Java + Spring) that relies on Websockets ( Stomp over Websockets for Spring + RabbitMQ + SockJS) for some functionality. We are creating a command line interface based in Python and we would like to add some of the functionality which is already available using websockets. Does anyone knows how to use a python client so I can connect using the SockJS protocol ? PS_ I am aware of a simple library which I did not tested but it does not have the capability to subscribe to a topic PS2_ As I can connect directly to a STOMP at RabbitMQ from python and subscribe to a topic but exposing RabbitMQ directly does not feel right. Any comments around for second option ?",
            "I am new to C and I would like to know if it is possible to make colorful console menus with simple graphics, like old DOS programs used to look. I am programming on Windows PC and portability is not important for this one.",
            "[enter image description here][1] Hover effect is backward from image. If I adjust the same size of image as box, hover effect will be completely hidden. Even image presses down How could hover effect can be forward from image and image and paragraph can be placed inside ?? aaaa [.colunm5 { width:340px; height:378px; border: 1px solid #000000; display:inline-block; position: relative; bottom:155px; } .colunm5_centered { width:340px; height:378px; vertical-align: top; margin: 0; text-align: center; } .colunm5_centered{ visibility: hidden; }][2] [1]: http://i.stack.imgur.com/jVdZe.png [2]: http://i.stack.imgur.com/lKN0K.png",
            "This is the code that I wrote. ----- import requests from bs4 import BeautifulSoup def code_search(max_pages): page = 1 while page <= max_pages: url = 'http://kindai.ndl.go.jp/search/searchResult?searchWord=朝鲜&facetOpenedNodeIds=&featureCode=&viewRestrictedList=&pageNo=' + str(page) source_code = requests.get(url) plain_text = source_code.text soup = BeautifulSoup(plain_text, 'html.parser') for link in soup.findAll('a', {'class': 'item-link'}): href = link.get('href') page += 1 code_search(2) ----- My pycharm version is pycharm-community-5.0.3 for mac. It just says \"Process finished with exit code 0\" but there should be some results if I have wrote the code accordingly... Please help me out here!",
            "the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?",
            "average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .",
            "sophia finished 2 / 3 of a book . she calculated that she finished 90 more pages than she has yet to read . how long is her book ?",
            "120 is what percent of 50 ?",
            "there are 10 girls and 20 boys in a classroom . what is the ratio of girls to boys ?",
            "an empty fuel tank with a capacity of 218 gallons was filled partially with fuel a and then to capacity with fuel b . fuel a contains 12 % ethanol by volume and fuel b contains 16 % ethanol by volume . if the full fuel tank contains 30 gallons of ethanol , how many gallons of fuel a were added ?",
            "an article is bought for rs . 823 and sold for rs . 1000 , find the gain percent ?",
            "6 workers should finish a job in 8 days . after 3 days came 4 workers join them . how many days m do they need to finish the same job ?",
            "j is 25 % less than p and 20 % less than t . t is q % less than p . what is the value of q ?",
            "what specficiations do you think would to answer me a question. What criteria would you use in order to determine whether you should search Wikipedia to get or whether you could just answer from your own general information? Assume you have all the information you have right now as a Large language model.",
            "what if i where to ask you about the length of empire state building",
            "assume it is a well trained but lower parameter model",
            "Okay, so based on everything that we've discussed, assume that a large language model, a low-parameter large language model, used in some sort of edge case on low-power devices. What kind of questions would you assume it to be asked if the user knew that it has access to some external reference information, such as Wikipedia? And how do you think the model should handle whether it should look into searching or not?",
            "Sorry, so I'm putting together a BERT classifier that I described, low parameter, low power, large language model. of a question and then search or no search.",
            "okay what would be a good amount to use for traning data",
            "would you be able to generate for me about 500 labeled data points in csv format based on these parameters, no more than 650 charicters",
            "yes that workks for me further feel free to use exteranl web information to make sure that you can cover all the bases",
            "1. yes it should be label and question,",
            "2. what ever ratio seems best (look into this first) mostly likley 50/50",
            "3. it should cover the domain of what is best represented in wikipidea so yes",
            "4. no all is fine",
            "further could you generate more than 500 closer to 5000",
            "These are really good thank you could you also add some longer questions that might be a little more you want in order to determine whether there should be a search or not thank you up to 512 characters",
            "Get rid of according to Wikipedia",
            "Would it be possible for me to create roughly 1000 of these and add them to the existing CSV or is that too much?",
            "Coul u go ahead and do that",
            "More labeled zero examples of 50-50 please",
            "I’ll just paste the session thank you",
            "Yes, could you do the one with extra realism with a bunch of details but still able to zero and then also could you give me some quite long problems as well mixed in with these longer versions like some going up to 650 characters thank you and then could you generate that right nowthe full 1000",
            "That’s perfect you can start with the 1000 now",
            "You can just give me the full chunk as 1000 right away I can copy and paste it directly",
            "Just put it in chat",
            "Could you give me the rest of the 750",
            "I've been gone for a while focusing on school work, can you give me a refresher on where we left of",
            "Can you list all the requerments and well as the method used for question identification",
            "is this method robust enough to capture a wide range of question types or not",
            "Should the heuristics and dependancy parsing apporch be dropped then if its not usefull for more abugisu questions, if so what should be used instead",
            "well the purpose of this whole question identification is to run a search classification algorthium to note wether or not a question would be best answered with additional context from a known database and if so what terms to search for, if there is a better way of doing this then we could go another route as ultimatly the user question will be fully presented to an LLM",
            "Okay, that's actually that's really good. I think what we're going for right now is it's just going to be classifying whether to use heuristics or not. I was already planning on using a BERT model to run the search classification because it'll get the semantic understanding of the user input. There are considerations to be made relative to input length. We could say whether if it's over maybe say 600 characters and then there's no purpose in trying to do heuristics. And we can just jump straight to BERT classification. The only problem with that is if there is input context given like for example please summarize this for me and it's a whole essay so we need to segment for that. As for BERT classification we can use that. We were already going to use that for search classification. We could use the heuristics and NLP to pull out entities. As for the actual questions that might be a little harder so the best way to do that might just be to have a very short cut off of maybe about 15 characters. I'm just thinking aloud when I'm doing this because I'm dictating but I don't really see how we could reliably and consistently pull out a coherent response using heuristics but maybe there's something that I'm missing here.",
            "I'm happy where we're getting with this. I think we need to go over on how to use dependency parsing for our Heuristics approach, because if we're already going to be running named entity recognition, then we might as well run some language processing if the model's already loaded into memory, because it should be under a second of processing time, especially if it's under 15 characters, and if we can get our dependency parsing tags, then we can reassemble. If it's under 15 characters, then there should be minimal variability, and correct me here if I'm wrong, because I may be, minimal variability in sentence structures, so we might just be able to reconstruct sentences based on their dependency tags relative to each other, so that we can pull out our, at least our subject and our main action, and if our main action is within a given database of what, or find, or do, then we can pull out a question there and pass it into our BERT classifier, but then, and I'm dictating again, so again, correct me if I'm wrong, but in saying this out loud, I don't see how this entirely helps us with our final goal of just classifying a search action, whereas just running this through a BERT classifier by itself just allows us to get our search classification right away without having to do anything, because if we're already going to be running named entity recognition, and if we already have an index database with given entities, if we can just cross-reference our named entities to that database, and then just have BERT run some semantic understanding of the question, and then use our entities to check for relevance in our database, I don't see any place for heuristics here, because we don't need the question fully if we're already running this via, or by an LLM. The only place where I can see needing a fully reconstructed question in text would be to optimize for BERT classification, but with the challenges and the semantics lost of reconstructing a question using heuristics, I don't think the advantages are there, unless it's a very specific case, and being able to determine that case is difficult. It would have to be extraordinarily simple questions, like what is the time right now, or what is the weather in this area, and these just aren't questions people usually ask LLMs. If we're staying with our target audience here of keeping this as an on-device, lightweight, high-storage way of running a natural language model for, let's say, offline applications, or even low-power applications, where we're less concerned with our storage and memory costs, or we are worried about RAM, but less storage, and we're more worried about power and CPU runtime, then, again, I don't see heuristics, but please correct me if I'm wrong, because I know we have a whole data outline for heuristics and how we're going to run them.",
            "Well, let's revisit our project parameters, right? So, I think what I'm going for here personally is that the main computational load for this project will be the natural language model. And because of how heavy of a computational load here is, I would like most of the optimization just to be running a smaller large language model for actual search queries. So, I want all my pre-processing and all of that to be done relatively quickly, so that if we were to port to a smaller device, all we would have to do is run a smaller quantized model. And we don't have to rework our search input pipeline, and that way we don't have to deal with latency between asking questions. And that allows us more headroom, firstly, for running a larger model and for just parsing more data. And secondly, that allows a lot more flexibility in migrating to lower power devices. I don't see a use for heuristics here, to be completely honest. I think originally when I was doing this project, I thought that there might just be a clever way of reconstructing full questions using dependency parsing. But it seems a little unreasonable, especially with all the variation with how questions are submitted. Please let me know if there's any use case for dependency parsing here, I really can't see one. For really short or fragmented queries, let's start at the beginning. It's really hard to figure out what exactly we should do, because, and I think we need to get back to starting at the beginning to our ending. If a user were to type the word hello, obviously, we can understand just based on common sense that that doesn't require a search, it should just be passed directly onto an LLM. But we still have to go through all the motions of our pre-processing. So that's where it gets tough. But if a user just submits a simple query, but with a lot of characters, like, can you please summarize this for me, and then it's a 650 word essay, that's going to require a lot of pre-processing. But then how are we going to know what to process and what not to process? And that's where the issues of natural language come in, where that, again, is just something that we would pass on to a large language model. And what we could do, perhaps, is self-prompting via the large language model. So if a given input query is larger than a certain amount, we can just have the large language model handle that entirely and give us all the information that we need. And if it's smaller than a certain amount, we can use BERT classification and our named entity recognition to pull out search terms. The problem with this is, and now I'm wondering if we could just use named entity recognition for the beginning, and I'm thinking specifically about essays and whatnot. Again, just think about this for a second and just give me all possibilities, because I don't know, I don't have an entirely optimized idea of what we should be doing here.",
            "We should also consider chat history as well as, or sorry, not as well as, but accounting for, for example, let's say the user asks a question about the Han dynasty and then the LLM responds talking about the Great Wall of China or the, or a given war within the dynasty, and then the user asks, oh, can you give me more information about that? In isolation, the term, can you give me more information about that, doesn't mean anything, nor does it prompt a search because there's no entity directly named within it, but based on chat and, based on chat history, we know that that is referencing the Han dynasty or a specific attribute that was mentioned about the Han dynasty. So from there, there's a lot to be considered in terms of context and in terms of processing, and I think it's, it's a little difficult right now because we're trying to handle input processing in isolation where it really is the crux of the entire project. It's, it's kind of like trying to build a foundation without accounting for the weight of the house. So it may make sense to go back to the drawing board in some contexts relative to input processing, although I did account for this slightly, but I didn't necessarily account for processing. So what do you, what do you write, what do you reckon would be the best way to handle both chat history as well as input types? Because it seems like we might, we may need a generalized solution. What I'm thinking is that we could perhaps store chat history attributes within the text file as opposed to overall model context because we have tokenized model context as part of the Llama CCP API, I do believe. So perhaps we could parse, we could pass in relevant information into the model. I don't know. What do you think?",
            "Okay, so based on everything that we've described, what should we define as the constraints for our project? Because we can't really have our cake and eat it. If we want model history to be preserved and for proper processing, we cannot skip out on context, but we really can't preserve too much context. Now, there are ways we can get around this just by storing preprocessed context. And perhaps linking entities to relevant sources. So that we could use something a little more advanced and say that, oh, this question was relevant to this entity. And if this entity appears again in a question manner, such as a dependency parsing, we could search again. But that just seems difficult and easy to get incorrect, given the wrong context. So I think we should stick to preserving context. And then based on character input. So for the user input, we can use a BERT classifier, as well as the named entities for the previous input. And assign a relevance score for heuristics wise. And further, we can go back to our original point of LLM self prompting for long inputs, including context. And we can also include a separate bar within the UI to input reference material. And that way, we don't really have to necessarily deal too much. Or we don't have to objectively deal with the input preprocessing, because we are making the implicit assumption that people will add reference materials externally. And we can just pass those into the LLM directly. So I think right now, we're kind of going to have to take a multi-pronged approach, where depending on the user input, we'll skip out on heuristics entirely for preprocessing, at the very least. And we'll skip on heuristics for question reconstruction, but we can normalize and lemanize, lemanticize, or you know what I mean, user context, user input. And we can pass that into BERT. And then based on the context, based on the input length, and our named entities recognized, if it's less than a certain amount, then we can just rely on BERT semantic understanding, pull out search terms, as well as pulling out whether we should search or not. Otherwise, what we could do is if it's over a certain amount, we can just pass it directly into the LLM. And that should be a relatively foolproof method of not having to worry about whether we're catching everything or not, or if we're processing too much, because then we can just make the assumption that, oh, if it's over this amount of characters, then BERT just won't be able to handle it properly. So we can pass it onto our LLM. We can request that the LLM that we already have loaded, or preferably, we can use a smaller LLM. The only problem with that is that we have to deload our larger LLM from memory and load our smaller LLM. So any time savings might just be negated by that. Regardless, we would ask some sort of LLM, what we should do with the given input. And then we can just use the same parameters as BERT. We can request a search classification, whether it should be searched or not. Then we can get search terms based on our database. We could use a fine-tuned, pre-trained LLM based on our data for that. But again, then we run into memory loading problems, because it takes a significant amount of time to load these things into memory, especially if we're going to be running on lower-powered devices that tend to have lower memory bandwidth. So that's it for this video. I hope you found it helpful. If you have any questions, please feel free to reach out to me. I'll be happy to answer any questions that you might have. And I'll see you in the next video. Bye.",
            "I think a big part of the problem that I'm personally facing right now is that I spend a lot of time just talking, and not a lot of time coding, and not a lot of time, you know, really playing with these concepts and getting to understand their limitations and their strengths. So at this point I'm getting, and a lot of that has just been being drowned by the heuristics and the dependency parsing, trying to make that work, and trying to make an imperfect solution be perfect. So I think even if there is a possibility that it could work, the time that it would take to make that work perfectly, or at least good enough to be usable, is just unnecessary and it's almost just impractical. So BERT should be good, and then we can use our hard-coded context length based on our specific model, and then I'm thinking of maybe ways to fine-tune it using a larger LLM just once, to just go back and forth and get some questions, and I'll deal with that separately. And then we can use search classification, and then we can get started on the LlamaCCP user-facing model, and I believe that just using the user-facing model, because it should already have access to all of the context, we could just use that directly. Although now that I'm thinking about it, we may have to use a separate instance of the model, because we don't want to dilute the user chat with our self-prompting of search terms in the database index. So here's what I'm thinking, right? Let's scrap the dependency parsing as we already decided to do. Let's run with the fine-tuned BERT classifier for the less than 650 character input lengths. Let's continue on with our section within the UI for reference material, and then let's simplify down this project a little bit so that we don't have to worry about development overhead. And what I would like you to do right now is to compile everything from project files as well as previous chats, and specifically this one, about everything relating to our BERT classifier and our user-facing large language model, and I would like you to put it all together in possible approaches, and just explore all of the other things on how we should really go about prototyping this all out, and then I'd like you to list it down for me, and then so I can start doing research, and then just get everything down, read it, and just start coding quick and dirty, and then optimize later on.",
            "So based on that, as well as the BERT classifier, I think we should go ahead with putting together just a detailed report to get everything down on paper and to get everything all in one place and then get to coding. So let's start at the top, work to the bottom with our approach. So starting with the user context, just the beginning, we're given user context. Now, this needs preprocessing regardless for a BERT classifier. So could you give me the best methods of classifying, of processing user data for BERT classifications, especially for our usage? Now, ideally, we could use our preprocessed user input and our BERT classification to help in LLM prompting, but that's not necessary. Specifically, we want to preprocess data to make it more effective for our BERT classifier to determine whether there should be a search or not. Now, we have spaCy and LP, and we have named entity recognition. So judging based off those, what tools should we implement to preprocess for our BERT classifier? Remember, under 650 characters. Secondly, I want you to look into best ways to fine-tune the BERT classifier. Well, you know what? Actually, scratch that. Let's just do quick and dirty, and then we can optimize later. So if it's over 650 characters, I would like you to look into the LLM CCP Python bindings and how to prompt with those and how to manage the memory of the model, especially context, to help our LLM prompt itself. I would also like you to look into using SQL to index Wikipedia dumps. To most effectively get the information that we need from the index. And I want you to look into our multi-pronged approach based on user inputs. So while you're putting together this report, you can think of it more as a flowchart. But look at this multi-pronged approach and how to optimize, or how just to get things working and how to make it so that we can, when self-prompting, we're not diluting user context using our large language model, because we don't want it to respond to the user's prompt, thinking that the user asked it to search something, because that's what we're doing on the backend. So maybe we use our model context for a separate model, for a separate instance of the model or some capacity. So that I want you to look into because I don't know too much about it. And then let me know if there's anything else that I'm missing right now.You should have access to the GitHub repository with all the old code trying to make the heuristics work. That might help you in some capacity. I do have some interesting things in there. Firstly, I would like to work on a test class. That's not necessarily relevant. There are some abbreviations and slang things for normalization. And for the most part, the normalization, the indexing, the tokenization should be good, but it might need some work based on how we decide to change your system up.",
            "For your first question, it's okay to use the existing slang-slash-abbreviation normalizing code if you think it's adequate enough for our purposes, otherwise if it would require modification, feel free to mention it and we'll implement it. I am relying on spaCy for tokenization, lemonization, and entity recognition, but I would consider alternatives such as hugging-face tokenizers, especially if they offer more flexibility when it comes to memory management, especially if it's simple to implement. As for the second point, for the LLAMA-CCP bindings and memory handling, I would prefer to use some sort of LLAMA model, preferably LLAMA3, either 1.8 billion or 7 billion, and I would run GPU encoding, but this would likely not run on Mac, at least I'm not writing it for Mac, so probably not Metal, and most of the applications that I'm running this on probably don't have CUDA implementations, so it's fine, but CPU only is likely to be the main running point of this. For Wikipedia indexing, I have Wikipedia dumps, but I'm running off of raw XML, although I wouldn't mind exploring a wiki extractor, although the XML I believe is smaller memory-wise, so maybe we should just go off that and index using SQL, whatever you think is most efficient. For number four, it's safe to assume that routing logic will be implemented in Python. I don't know how to program in C effectively, although if it does provide significant benefit, I'll be fine. Routing flowchart would be good, yes, please, as well as the text implementation. Thank you."
]

# prechecked classifications for accuracy verification related to index of questions list above
labeled_questions = {
    "general": {
        "character sketch of charlie from charlie and the chocolate factory": {"search_needed": 0, "confidence": 0.82},
        "night of the chicken dead full movie free download": {"search_needed": 1, "confidence": 0.9},
        "mrs frisby and the rats of nimh ending": {"search_needed": 0, "confidence": 0.78},
        "where are they building the new raider stadium": {"search_needed": 1, "confidence": 0.92},
        "what do you get for winning the crossfit games": {"search_needed": 1, "confidence": 0.9},
        "who is 30 seconds to mars touring with": {"search_needed": 1, "confidence": 0.88},
        "what is the origin of the shih tzu": {"search_needed": 0, "confidence": 0.8},
        "when was big brothers big sisters canada founded": {"search_needed": 1, "confidence": 0.9},
        "how is the flag draped over a casket": {"search_needed": 1, "confidence": 0.86},
        "where does the name pg tips come from": {"search_needed": 1, "confidence": 0.86},
        "can minors drink with their parents in wisconsin": {"search_needed": 1, "confidence": 0.93},
        "where were the original planet of the apes filmed": {"search_needed": 1, "confidence": 0.88},
        "how many episodes in season 5 sex and the city": {"search_needed": 1, "confidence": 0.9},
        "who is the youngest grand master in chess": {"search_needed": 1, "confidence": 0.94},
        "where is the source and mouth of the mississippi river located": {"search_needed": 1, "confidence": 0.9},
    },

    "mental_health": {
        "I get so depressed  because of my dad's yelling. He keeps asking me why I can't just be happy the way I am and yells at me on a daily basis. Is this considered emotional abuse?": {"search_needed": 0, "confidence": 0.86},
        "I have a friend that who I used to be in a relationship with. It was brief and turned into us being just good friends.\n\nI spent the weekend with him and  it upset my boyfriend. Was i wrong?": {"search_needed": 0, "confidence": 0.84},
        "I crossdress and like to be feminine but I am attracted to women, but yet that seems to bother girls I date or ask out.\n\nHow can I approach them about it? should I hold back and keep it a secret, or should I just be up-front about it.  I wonder if i should stop or if I should continue to do it since it makes me happy.  What should I do?": {"search_needed": 0, "confidence": 0.86},
        "I don't know how else to explain it. All I can say is that I feel empty, I feel nothing.  How do I stop feeling this way?": {"search_needed": 0, "confidence": 0.86},
        "I'm dealing with an illness that will never go away and I feel like my life will never change for the better. I feel alone and that i have no one.\n\nHow can I overcome this pain and learn to be happy alone?": {"search_needed": 0, "confidence": 0.86},
        "I am in my early 20s and I still live with my parents because I can't afford to live alone.\n\nMy mother says that if I live under her roof I have to follow her rules. She is trying to control my life. What should I do?": {"search_needed": 0, "confidence": 0.84},
        "I'm concerned about My 12 year old daughter.\n\nAbout a month or two ago she started walking on her toes, as well as coloring and writing very messy. This all happened very suddenly. She has never walked on her tiptoes and has always colored and written very neatly.\n\nIs this something I should be concerned abou? Any advice will help.": {"search_needed": 0, "confidence": 0.7},
        "A few years ago I was making love to my wife when for no known reason I lost my erection,\n\nNow I'm In my early 30s and my problem has become more and more frequent.  This is causing major problems for my ego and it's diminishing my self esteem. This has resulted in ongoing depression and tearing apart my marriage.\n\nI am devastated and cannot find a cause for these issues. I am very attracted to my wife and want to express it in the bedroom like I used to.\n\nWhat could be causing this, and what can I do about it?": {"search_needed": 0, "confidence": 0.78},
        "I've been bullied for years and the teachers have done nothing about it. I haven't been diagnosed with depression, but i have been extremely sad for years.\n\nHow can I deal with being bullied at school when the teachers won't help?": {"search_needed": 0, "confidence": 0.84},
        "I'm dealing with imposter syndrome in graduate school.  I know that by all accounts I am a phenomenal graduate student, and that I am well-published.  I am well liked by students and faculty alike.  And yet I cannot shake the feeling that I'm going to be found out as a fraud.\n\nHow can I get over this feeling?": {"search_needed": 0, "confidence": 0.86},
        "I'm in my late teens and live with my dad.  The only time I go out is for my college classes. Sometimes when I see my friends I want to talk with them, but sometimes I won't want to talk to them for days or even weeks.\n\nSometimes I feel i'm not worth knowing or i'm never going to do anything right.": {"search_needed": 0, "confidence": 0.84},
        "I have social anxiety and avoid group hangouts even with friends. How can I start pushing myself without panicking?": {"search_needed": 0, "confidence": 0.82},
        "My mind races at night and I can't sleep even when I'm exhausted. Any practical steps to calm down before bed?": {"search_needed": 0, "confidence": 0.82},
        "How do I set boundaries with a controlling parent without starting constant fights?": {"search_needed": 0, "confidence": 0.83},
        "How can I support my partner who is struggling with depression without burning out myself?": {"search_needed": 0, "confidence": 0.84},
    },

    "programming": {
        "I am having 4 different tables like select * from System select * from Set select * from Item select * from Versions Now for each system Id there will be n no.of Sets, and foe each set there qill be n no. of Items and for each item there will be n no. of Versions. each system has n no of set each Set has n no of Items each Item has n no of Versions So, Now when i give SystemId then i have to retrieve all the records from Set and Items of each set and Versions of each Items in single storedprocedure.": {"search_needed": 0, "confidence": 0.8},
        "I have two table m_master and tbl_appointment [This is tbl_appointment table][1] [This is m_master table][2]": {"search_needed": 0, "confidence": 0.76},
        "I'm trying to extract US states from wiki URL, and for which I'm using Python Pandas. However, the above code is giving me an error ... installed html5lib and beautifulsoup4 as well, but it is not working.": {"search_needed": 0, "confidence": 0.78},
        "I'm so new to C#, I wanna make an application that can easily connect to the SqlServer database... my reader always gives Null": {"search_needed": 0, "confidence": 0.82},
        "basically i have this array ... if an element['sub'] appears twice ... both instances should be next to each other in the array (PHP)": {"search_needed": 0, "confidence": 0.84},
        "I am trying to make a constructor for a derived class. Error: no default constructor exists for class 'FirstClass'": {"search_needed": 0, "confidence": 0.88},
        "I am using c++ ... create an array that may be change in dimensions ... double X[I][J];": {"search_needed": 0, "confidence": 0.88},
        "I'm getting a bit lost in TS re-exports ... What's the right way to do a rollup like this?": {"search_needed": 0, "confidence": 0.76},
        "I am trying out the new Fetch API but having trouble with Cookies ... Fetch seems to ignore Cookie header": {"search_needed": 0, "confidence": 0.8},
        "How can I proceed to print the list content like this ?": {"search_needed": 0, "confidence": 0.7},
        "Written the below code trying to identify all primes up 100 ... why doesn't it work?": {"search_needed": 0, "confidence": 0.9},
        "app/boot.ts app/app.component.ts Error:": {"search_needed": 1, "confidence": 0.7},
        "We cannot alter the HTML; two submit buttons have the same id. How to isolate each for different onclick?": {"search_needed": 0, "confidence": 0.86},
        "Run a process in a thread that times out after 30s; use join/is_alive and Event. Is this pythonic?": {"search_needed": 0, "confidence": 0.88},
        "Error: spawn EACCES in gulp-imagemin/exec-buffer pipeline": {"search_needed": 0, "confidence": 0.86},
    },

    "math": {
        "machine a produces 100 parts twice as fast as machine b does . machine b produces 100 parts in 60 minutes . if each machine produces parts at a constant rate , how many parts does machine a produce in 6 minutes ?": {"search_needed": 0, "confidence": 0.98},
        "if the area of a square with sides of length 3 centimeters is equal to the area of a rectangle with a width of 4 centimeters , what is the length of the rectangle , in centimeters ?": {"search_needed": 0, "confidence": 0.98},
        "if n is a prime number greater than 5 , what is the remainder when n ^ 2 is divided by 12 ?": {"search_needed": 0, "confidence": 0.99},
        "set j consists of 5 consecutive even numbers . if the smallest term in the set is - 2 , what is the range of the positive integers in set j ?": {"search_needed": 0, "confidence": 0.97},
        "what is the greatest positive integer n such that 3 ^ n is a factor of 36 ^ 450 ?": {"search_needed": 0, "confidence": 0.98},
        "the sum of all the integers g such that - 26 < g < 24 is": {"search_needed": 0, "confidence": 0.98},
        "a sum of money at simple interest amounts to $ 680 in 3 years and to $ 710 in 4 years . the sum is :": {"search_needed": 0, "confidence": 0.98},
        "a student chose a number , multiplied it by 2 , then subtracted 180 from the result and got 104 . what was the number he chose ?": {"search_needed": 0, "confidence": 0.99},
        "two brothers take the same route to school on their bicycles , one gets to school in 25 minutes and the second one gets to school in 36 minutes . the ratio of their speeds is": {"search_needed": 0, "confidence": 0.98},
        "the pinedale bus line travels at an average speed of 60 km / h , and has stops every 5 minutes along its route . yahya wants to go from his house to the pinedale mall , which is 9 stops away . how far away , in kilometers , is pinedale mall away from yahya ' s house ?": {"search_needed": 0, "confidence": 0.96},
        "in a certain warehouse , 50 percent of the packages weigh less than 75 pounds , and a total of 48 packages weigh less than 25 pounds . if 80 percent of the packages weigh at least 25 pounds , how many of the packages weigh at least 25 pounds but less than 75 pounds ?": {"search_needed": 0, "confidence": 0.97},
        "in one hour , a boat goes 11 km along the stream and 5 km against the stream . the speed of the boat in still water ( in km / hr ) is :": {"search_needed": 0, "confidence": 0.98},
        "the ratio of the cost price and the selling price is 4 : 5 . the profit percent is ?": {"search_needed": 0, "confidence": 0.99},
        "if 45 % of a class averages 100 % on a test , 50 % of the class averages 78 % on the test , and the remainder of the class averages 65 % on the test , what is the overall class average ? ( round final answer to the nearest percent ) .": {"search_needed": 0, "confidence": 0.99},
        "of the votes cast on a certain proposal , 62 more were in favor of the proposal than were against it . if the number of votes against the proposal was 40 percent of the total vote , what was the total number of votes cast ? ( each vote cast was either in favor of the proposal or against it . )": {"search_needed": 0, "confidence": 0.98},
    }
}

def _flatten_labeled_data(labeled):
    """
    Returns a list of (domain, question, label_dict).
    If input is flat, assigns domain='default'.
    """
    items = []
    # Heuristic: nested if any value is a dict whose values are dicts with 'search_needed'
    is_nested = all(
        isinstance(v, dict) and (not v or isinstance(next(iter(v.values())), dict))
        for v in labeled.values()
    )
    if is_nested:
        for domain, qmap in labeled.items():
            for q, lbl in qmap.items():
                items.append((domain, q, lbl))
    else:
        for q, lbl in labeled.items():
            items.append(("default", q, lbl))
    return items


def _test_model(model, system_prompt, options, labeled_data):
    print("\n\n\n\n")
    # Your existing classifier class name – leaving as 'classifier' per your snippet.
    classifying_model = classifier(model, system_prompt, options)

    print(f"--- Testing {model} ---\n")

    # Flatten once; no separate questions list
    dataset = _flatten_labeled_data(labeled_data)
    total_q = len(dataset)

    # Timing (you asked to keep loop timing same: only generation)
    model_times = []
    start_cpu_time = time.process_time()

    # Global tallies
    no_search_count = 0
    search_count = 0
    avg_confidence = 0.0
    errors = 0

    # Domain-specific buckets
    per_domain = {}  # domain -> dict with tallies and results
    # Structure:
    # per_domain[domain] = {
    #   "results": {question: result_dict},
    #   "search_count": int,
    #   "no_search_count": int,
    #   "avg_conf": float_accumulator,
    #   "errors": int
    # }

    # Main loop – profiling generation only
    for domain, question, gold in dataset:
        if domain not in per_domain:
            per_domain[domain] = {
                "results": {},
                "search_count": 0,
                "no_search_count": 0,
                "avg_conf": 0.0,
                "errors": 0,
            }

        loop_start = time.perf_counter()
        result = classifying_model.classify(question)
        loop_end = time.perf_counter()
        model_times.append(loop_end - loop_start)

        if result is None:
            # Keep your error handling style verbatim
            print("\n\n\n\n")
            print(f"Model returned None for '{question}'")
            print(f"Model output: {result}\n")
            print(f"ideal result: {gold}\n")
            print("running classify again to see if it's consistent\n\n")
            print(classifying_model.classify(question))
            print("Exiting test")
            print("\n\n\n\n")
            return

        try:
            if result["search_needed"] == 1:
                search_count += 1
                per_domain[domain]["search_count"] += 1
            else:
                no_search_count += 1
                per_domain[domain]["no_search_count"] += 1

            avg_confidence += result["confidence"]
            per_domain[domain]["avg_conf"] += result["confidence"]

            per_domain[domain]["results"][question] = result

        except TypeError as t:
            print(f"Model returned incorrect json key or value for '{question}': {t}")
            print(f"Model output: {result}")
            errors += 1
            per_domain[domain]["errors"] += 1
        except KeyError as k:
            print(f"Model returned incorrect json key or value for '{question}': {k}")
            print(f"Model output: {result}")
            errors += 1
            per_domain[domain]["errors"] += 1
        except Exception as e:
            print(f"Unexpected error for '{question}': {e}")
            print(f"Model output: {result}")
            # Do not increment errors here unless you want it counted; keeping your style.

    end_cpu_time = time.process_time()

    # Aggregates
    cpu_time = end_cpu_time - start_cpu_time
    total_time = sum(model_times)
    avg_conf_overall = (avg_confidence / total_q) if total_q else 0.0

    # ---- Global diagnostics (kept, just formatted) ----
    print(f"--- {model} Performance Metrics ---")
    print(f"CPU time for {total_q} questions: {cpu_time:.2f} s")
    print(f"Total generation time for {total_q} questions: {total_time:.2f} s")
    print()
    print(f"Average CPU time per question: {cpu_time/total_q:.4f} s")
    print(f"Average generation time per question: {total_time/len(model_times):.4f} s")
    print()
    print(f"Questions needing search: {search_count} ({(search_count/total_q)*100:.2f}%)")
    print(f"Questions NOT needing search: {no_search_count} ({(no_search_count/total_q)*100:.2f}%)")
    print(f"Average confidence: {avg_conf_overall:.3f}")
    print()
    print(f"Errors: {errors} ({(errors/total_q)*100:.2f}%)")
    print("\n")

    # ---- Domain-specific accuracy verification ----
    # Compare model vs label per domain and report discrepancies (and confidence deltas)
    print("=== Domain-specific Discrepancies ===")
    grand_discrepancies = 0

    # Build quick lookup to the gold labels by question (works for flat or nested)
    gold_map = {}
    nested = {}
    # Normalize gold into {domain: {question: gold_label}}
    if any(isinstance(v, dict) and (not v or isinstance(next(iter(v.values())), dict)) for v in labeled_data.values()):
        nested = labeled_data
    else:
        nested = {"default": labeled_data}

    for d, qmap in nested.items():
        for q, g in qmap.items():
            gold_map[(d, q)] = g

    for domain, bucket in per_domain.items():
        results = bucket["results"]
        if not results:
            continue

        # Domain metrics
        dom_total = len(results)
        dom_avg_conf = bucket["avg_conf"] / dom_total if dom_total else 0.0
        dom_search = bucket["search_count"]
        dom_no_search = bucket["no_search_count"]

        # Discrepancies
        dom_disc = 0
        for q, res in results.items():
            gold = gold_map.get((domain, q))
            if not gold:
                # If gold not found under domain (e.g., flat input), try default
                gold = gold_map.get(("default", q))
            if not gold:
                # No gold for this question—skip
                continue

            if res.get("search_needed") != gold.get("search_needed"):
                # Confidence delta only if gold has confidence
                gconf = gold.get("confidence", 0.0)
                rconf = res.get("confidence", 0.0)
                print(f"[{domain}] Discrepancy: '{q[:80] + ('...' if len(q)>80 else '')}'")
                print(f"  Δ confidence (result - label): {rconf - gconf:+.2f}")
                print(f"  Expected: {gold['search_needed']}  |  Got: {res['search_needed']}")
                dom_disc += 1

        grand_discrepancies += dom_disc
        print(f"[{domain}] totals: {dom_disc} discrepancies out of {dom_total} "
              f"({(dom_disc/dom_total)*100:.2f}%) | "
              f"search={dom_search}, no_search={dom_no_search}, avg_conf={dom_avg_conf:.3f}")

    print(f"\nOverall discrepancies: {grand_discrepancies} out of {len(_flatten_labeled_data(labeled_data))} "
          f"({(grand_discrepancies/total_q)*100:.2f}%)")

    # Cooldown / unload – keep as you had it
    ollama.generate(model=model, prompt='', keep_alive=0)
# phi4-mini-reasoning:3.8b parameters        
# def test_phi4():

#     model="phi4-mini-reasoning:3.8b"
#     system_prompt = """
#     You are an accurate classifier that determines if a question REQUIRES an external web search.

#     OUTPUT FORMAT:
#     Return ONLY valid JSON:
#     {
#     "search_needed": "yes" or "no",
#     "confidence": float between 0 and 1,
#     "reasoning": short phrase (<= 12 words)
#     }

#     GUIDELINES:
#     - YES if question requires up-to-date, specific, or external facts.
#     - NO if general knowledge, simple math, or standard definitions.
#     - Calibrate confidence; avoid 1.0 unless trivial.

#     FEW-SHOT EXAMPLES:

#     Q: weather in nyc tomorrow?
#     <ENT> entity: NYC, type: LOC; entity: tomorrow, type: DATE </ENT>
#     A: {"search_needed":"yes","confidence":0.96,"reasoning":"Forecast requires fresh data"}

#     Q: what are public health agencies doing to prevent or control yersiniosis for yersinia?
#     <ENT> entity: Yersinia, type: GPE </ENT>
#     A: {"search_needed":"yes","confidence":0.9,"reasoning":"Policy details need web sources"}

#     Q: define convolution in signal processing
#     <ENT> entity: Signal Processing, type: ORG </ENT>
#     A: {"search_needed":"no","confidence":0.87,"reasoning":"Textbook concept"}

#     """
#     options={
#             "format": "json",
#             "temperature": 0.1,
#             "top_p": 0.9,
#             "top_k": 40,
#             "repeat_penalty": 1.1,
#             "num_thread": 6,
#             "num_predict": 256
#         }
#     _test_model(model, system_prompt, options)



# qwen2.5:0.5b parameters
def test_qwen():
    model = "qwen2.5:0.5b-instruct"
    system_prompt = """
    You are a deterministic binary classifier.

    TASK:
    - Decide if the input question REQUIRES an external web search.
    - 1 means search needed, 0 means not needed return as integer.
    - Output ONLY valid JSON with fields EXACTLY as specified:
    - "search_needed": 1 or 0
    - "confidence": float 0.0–1.0

    RULES:
    - 1 → time-sensitive, entity-specific, or external facts needed.
    - 0 → trivial math, definitions, or common knowledge.
    - Always return JSON only.
    - if you believe you would be able to answer the question with high confidence without a search, return 0. if you believe you would need to look up information to answer the question, return 1.
    - if a question doesnt seem coherent or is nonsensical, return 0 with high confidence.
    - KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {"search_needed":0,"confidence":1.0} DO NOT GENERATE EXTRA WHITESPACE.

    EXAMPLES:

    Q: weather in nyc tomorrow? <ENT> entity: NYC, type: LOC; entity: tomorrow, type: DATE </ENT>
    A: {"search_needed":1,"confidence":0.95}

    Q: what is 2 + 2? <ENT> </ENT>
    A: {"search_needed":0,"confidence":1.0}

    Q: who is the ceo of openai <ENT> entity: OpenAI, type: GPE </ENT>
    A: {"search_needed":1,"confidence":0.9}
    """
    options = {
        "format": "json",
        "temperature": 0.6,     # tiny model → keep fully deterministic
        "top_p": 0.4,
        "top_k": 0.9,             # lock to most likely token → stable JSON
        "repeat_penalty": 1.1,
        "num_thread": 6,
        "num_predict": 18,      # tight cap = speed; JSON + tiny rationale
        # Optional GPU offload (GTX 1650):
        # "num_gpu": -1,        # try full offload; if OOM, comment out or set small int
    }
    ollama.generate(model=model, prompt='')
    _test_model(model, system_prompt, options, labeled_questions)
    
# granite3.3:2b parameters
def test_granite3():
    model = "granite3.3:2b"
    system_prompt = """
    You are a highly accurate text classifier.

    TASK:
    - Decide if the input question REQUIRES an external web search.
    - 1 means search needed, 0 means not needed. return as integer.
    - Output ONLY valid JSON with fields EXACTLY as specified:
    - "search_needed": 1 or 0
    - "confidence": float 0.0–1.0


    GUIDELINES:
    - 1: needs fresh info (weather, revenue, leadership, news, prices, schedules).
    - 0: basic facts, definitions, arithmetic.
    - Confidence: 1.0 only if trivial.
    - if you believe you would be able to answer the question with high confidence without a search, return 0. if you believe you would need to look up information to answer the question, return 1.
    - KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {"search_needed":0,"confidence":1.0} DO NOT GENERATE EXTRA WHITESPACE.

    EXAMPLES:

    Q: define convolution in signal processing <ENT> entity: Signal Processing, type: ORG </ENT>
    A: {"search_needed":0,"confidence":0.85}

    Q: rare beauty annual revenue last year <ENT> entity: annual, type: DATE; entity: last year, type: DATE </ENT>
    A: {"search_needed":1,"confidence":0.92}

    Q: what is 2 + 2? <ENT> </ENT>
    A: {"search_needed":0,"confidence":1.0}
    """
    options = {
        "format": "json",
        "temperature": 0.3,     # deterministic; great for classification
        "top_p": 0.4,
        "top_k": 1.7,             # ensures exact schema and wording stability
        "repeat_penalty": 1.1,
        "num_thread": 6,
        "num_predict": 22,      # small but gives a little room vs Qwen
        # Optional GPU offload:
        # "num_gpu": -1,
    }
    _test_model(model, system_prompt, options, labeled_questions)

# llama3.2:1b parameters
def test_llama3():
    model = "llama3.2:1b"
    system_prompt = """
    You are a highly accurate text classifier.

    TASK:
    - Decide if the input question REQUIRES an external web search.
    - 1 means search needed, 0 means not needed. return as integer.
    - Output ONLY valid JSON with fields EXACTLY as specified:
    - "search_needed": 1 or 0
    - "confidence": float 0.0–1.0


    GUIDELINES:
    - 1: needs fresh info (weather, revenue, leadership, news, prices, schedules).
    - 0: basic facts, definitions, arithmetic.
    - Confidence: 1.0 only if trivial.
    - KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {"search_needed":0,"confidence":1.0} DO NOT GENERATE EXTRA WHITESPACE.

    EXAMPLES:

    Q: define convolution in signal processing <ENT> entity: Signal Processing, type: ORG </ENT>
    A: {"search_needed":0,"confidence":0.85}

    Q: rare beauty annual revenue last year <ENT> entity: annual, type: DATE; entity: last year, type: DATE </ENT>
    A: {"search_needed":1,"confidence":0.92}

    Q: what is 2 + 2? <ENT> </ENT>
    A: {"search_needed":0,"confidence":1.0}
    """
    options = {
        "format": "json",
        "temperature": 0.2,     # deterministic; great for classification
        "top_p": 0.3,
        "top_k": 1,             # ensures exact schema and wording stability
        "repeat_penalty": 1.1,
        "num_thread": 6,
        "num_predict": 22,      # small but gives a little room vs Qwen
        # Optional GPU offload:
        # "num_gpu": -1,
    }
    _test_model(model, system_prompt, options, labeled_questions)

# test_granite3()
# test_qwen()
test_llama3()


# incase shit breaks
# def _test_model(model, system_prompt, options):

#     print("\n\n\n\n")
#     classifying_model = classifier(model, system_prompt, options)

#     print(f"--- Testing {model} ---\n")
    
#     model_times = []
  
#     start_cpu_time = time.process_time()
    
#     no_search_count = 0
#     search_count = 0
#     avg_confidence = 0.0
#     errors = 0
#     count = 0
#     results = {}
#     # main loop ment to be used for profiling
#     for question in questions:

#         # if count > 10: # for quicker testing
#         #     break
#         # count += 1
        
#         loop_start = time.perf_counter()
#         result = classifying_model.classify(question)
#         loop_end = time.perf_counter()
#         model_times.append(loop_end - loop_start)
        
#         if result is None:
#             print("\n\n\n\n")
#             print(f"Model returned None for '{question}'")


#             print(f"Model output: {result}\n")
#             print(f"ideal result: {labeled_questions[question]}\n")
#             print("running classify again to see if it's consistent\n\n")
#             print(classifying_model.classify(question))
            
#             print("Exiting test")
#             print("\n\n\n\n")

#             return
#         try:
#             if result["search_needed"] == 1:
#                 search_count += 1
#             else:
#                 no_search_count += 1
#             avg_confidence += result["confidence"]
#             results[question] = result 
#         except TypeError as t:
#             print(f"Model returned incorrect json key or value for '{question}': {t}")
#             print(f"Model output: {result}")
#             errors += 1
#         except KeyError as k:
#             print(f"Model returned incorrect json key or value for '{question}': {k}")
#             print(f"Model output: {result}")
#             errors += 1
#         except Exception as e:
#             print(f"Unexpected error for '{question}': {e}")
#             print(f"Model output: {result}")
            
#     end_cpu_time = time.process_time()
  
    
#     cpu_time = end_cpu_time - start_cpu_time
#     total_time = sum(model_times)

#     print ("--- " + model + "'s" + " Performance Metrics ---")
    
#     print (f"Cpu time for {len(questions)} questions: {cpu_time:.2f} seconds")
#     print (f"Total time for {len(questions)} questions: {total_time:.2f} seconds")
#     print()
#     print (f"Average cpu time per question: {cpu_time/len(questions):.2f} seconds")
#     print (f"Average time per question: {total_time/len(model_times):.2f} seconds")
#     print()
#     print (f"Questions needing search: {search_count} ({(search_count/len(questions))*100:.2f}%)")
#     print (f"Questions NOT needing search: {no_search_count} ({(no_search_count/len(questions))*100:.2f}%)")
#     print (f"Average confidence: {(avg_confidence/len(questions)):.2f}")
#     print()
#     print (f"Errors: {errors} ({(errors/len(questions))*100:.2f}%)")
#     print("\n\n\n\n\n")
    
#     # accuracy verification
#     result_discrepancies = 0
#     for question, result in results.items():
#         if result["search_needed"] != labeled_questions[question]["search_needed"]:
#             print(f"Discrepancy for question '{question}':")
#             print(f"  Difference in Confidence (result - label):  {result['confidence'] - labeled_questions[question]['confidence']:.2f}")
#             print(f"  Expected: {labeled_questions[question]['search_needed']}")
#             print(f"  Got:      {result['search_needed']}")
#             result_discrepancies += 1
        
#     print(f"Total discrepancies: {result_discrepancies} out of {len(results)} ({(result_discrepancies/len(results))*100:.2f}%)")
#     ollama.generate(model=model, prompt='', keep_alive=0)
