import pandas as pd
import numpy as np
from transformers import pipeline
url="https://drive.google.com/file/d/17_HRkT6T6LnGHYDDKG-NNDPenTW-vMbC/view?usp=sharing"
url='https://drive.google.com/uc?id='+url.split('/')[-2]

## Скачивание датасета с гугл диска

DATA = pd.read_csv(url)
kol_dial=np.array(DATA['dlg_id'])[-1]
dialogs= [ DATA[DATA['dlg_id']==i] for i in range(kol_dial+1) ]

## Выбор НС и режима ее работы

p = pipeline(
  task='zero-shot-classification', 
  model='cointegrated/rubert-base-cased-nli-twoway'
)

## Функция для обработки 1 предложения:
## предложение режется по словам и получаем ответ от НС от каждого кусочка
##

def sent_processing(sent,param):
  #plt.figure(figsize=(40,20))
  result=[]
  data=[ sent.split(' ')[:i+1] for i in range(len(sent.split(' '))) ]

  ## Actions - набор "смысловых нагрузок в предложении" ,
  # как результат работы НС получаем результат по каждому "смыслу"

  if len(data)>3:
    actions=['здравствуйте', 'до свидания', 'имя', 'название компании']
  else:
    actions=['здравствуйте', 'до свидания', 'имя человека', 'название компании']
  for act in actions:
    graph=[]
    
    for t in data:
      qwerty=''
      for word in t:
        qwerty+=' '+word
      graph.append(p(    sequences=qwerty, 
                          candidate_labels=act, 
                          hypothesis_template="это {}.")['scores'][0])
    #plt.scatter(range(len(graph)),graph, label=act)
    if max(graph)>param:
      indx=graph.index(max(graph))
      if act=='имя' or act=='название компании':
        result.append({act: sent.split(' ')[indx]})
      elif act=='здравствуйте':
        result.append('приветствие')
      else:
        result.append('прощание')
  #plt.legend()
  #plt.plot()
  
  return result


i=0
RESULTS=[]  # Список ответов для каждого предложения



for dil in dialogs:
  for row in np.array(dil):
    result=[]
    if row[2]=='manager':
      sent=row[3]
      result.append(sent_processing(sent,param=0.85))
    RESULTS.append(result)
      #print(actions[answer.index(max(answer))])


DATA['ВЫВОД']=RESULTS   # Добавления столбца с результатами в исходную таблицу



name=f"Таблица с результатами.xlsx"

# Сохранение полученно таблицы в Excel формате

DATA.to_excel(name)
print('По результатам анализа: ')
ind=0

# Проверка условия, что менеджер поздоровался и попрощался

for n_d in range(len(dialogs)):
  privet=False
  dosvid=False
  for i in range(len(dialogs[n_d])):
    #print(RESULTS[ind])
    if ['приветствие'] in RESULTS[ind]:
      privet=True
    if ['прощание'] in RESULTS[ind]:
      dosvid=True
    ind+=1
  if privet and dosvid:
    print(f'{n_d} менеджер выполнил условие')
