import fasttext
import pandas as pd

m=fasttext.load_model("/home/model_dictionary_doYouMean.bin")

#input text
texts='''

Statement of Purpose

I am writing this statement of purpose to express my strong interest in pursuing a graduate program in Social Work in the United States. With a bachelor's degree in economics, work experience in human resources, and voluntary work in an NGO, I have developed a diverse background that has fueled my passion for making a positive impact on underprivileged communities. I believe that a higher degree in social work will provide me with the knowledge, skills, and opportunities to contribute meaningfully to the lives of those in need.

During my undergraduate studies in economics, I gained a deep understanding of the socioeconomic factors that contribute to social inequalities. I recognized that economic policies and practices have a direct impact on marginalized populations, often exacerbating their already challenging circumstances. This realization sparked my desire to explore social work as a means to address the underlying causes of social injustices and work towards creating a more equitable society.

My professional experience in human resources has allowed me to witness firsthand the challenges faced by individuals in the workforce, particularly in terms of diversity, equity, and inclusion. I have had the privilege of working with employees from various backgrounds, understanding their unique needs, and implementing strategies to foster an inclusive and supportive work environment. This experience has further solidified my commitment to social justice and advocacy, as I have witnessed the transformative power of creating environments that prioritize the well-being and empowerment of individuals.

In addition to my professional experience, I have actively engaged in voluntary work with an NGO. This experience has been incredibly fulfilling, as it has provided me with the opportunity to directly assist underprivileged individuals and communities. Through my work with the NGO, I have developed a deep sense of empathy, cultural sensitivity, and the ability to build meaningful connections with diverse groups of people. This experience has affirmed my belief in the transformative potential of social work and has motivated me to pursue a higher degree in this field.

I am particularly drawn to the social work graduate programs in the United States due to their strong emphasis on academic rigor, practical training, and research. I am eager to deepen my understanding of social welfare policies, community development, and advocacy, and to explore innovative approaches to address poverty, inequality, and discrimination. I am keen to learn from renowned faculty members who are experts in the field and collaborate with fellow students who share a similar passion for social justice.

Furthermore, I am excited about the experiential learning opportunities offered by these programs, such as field placements and internships. These practical experiences will allow me to apply my theoretical knowledge to real-world situations, gain valuable hands-on skills, and further develop my ability to work effectively with diverse populations. I am eager to contribute to the well-being of individuals and communities by implementing evidence-based interventions, advocating for policy changes, and fostering sustainable community development.

In conclusion, my background in economics, human resources, and voluntary work in an NGO has shaped my desire to pursue a graduate program in Social Work in the United States. I am passionate about addressing social injustices, empowering marginalized populations, and creating positive change in society. I am confident that a higher degree in social work will equip me with the necessary knowledge, skills, and experiences to make a meaningful difference in the lives of those in need. I am excited about the opportunity to learn, grow, and contribute to the field of social work, and I look forward to the challenges and rewards that lie ahead.

'''

texts=texts.replace('\n',''). replace ('\r','')

label,prob=m.predict(texts,k=53)



doYouOffering=[]

doYouOffering = pd.DataFrame({'interests':label})

doYouOffering['probability'] = pd.DataFrame({'probability':prob})


# clearance zones prefix and suffix 
doYouOffering['interests'] = doYouOffering['interests'].str.replace('__label__','')

doYouOffering['interests'] = doYouOffering['interests'].str.replace(',"To','')

print (doYouOffering)
