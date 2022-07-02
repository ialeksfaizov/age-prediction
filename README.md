# Определение возраста покупателей

## Условный сетевой супермаркет внедряет систему компьютерного зрения для обработки фотографий покупателей. Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы:
- Анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы;
- Контролировать добросовестность кассиров при продаже алкоголя.

Необходимо построить модель, которая по фотографии определит приблизительный возраст человека. В распоряжении набор фотографий людей с указанием возраста.

## Stack
- Pandas
- Numpy
- Seaborn
- Statsmodels
- PIL 
- TensorFlow
- Keras
- ResNet

## Исходные данные содержат:

- `7591` изображение;
- данные чистые, без явных дубликатов;
- данные распределены нормально со скошенностью вправо;
- наблюдается бимодальность - после `30` лет наблюдаются пики с интервалом в 10 лет;
![customer-age-prediction (1)](https://user-images.githubusercontent.com/94479037/176999388-c141d312-7cda-4c76-936f-92c167d3c164.png)
- наблюдаются выбросы после `70` лет;
![customer-age-prediction (2)](https://user-images.githubusercontent.com/94479037/176999395-9a0e0ef9-245e-47c1-a5bf-cdacc5694244.png)
- большинство данных собраны на людях с возрастом `30` лет;
- изображения имеют разные размеры - чаще всего изображения имеют размер `517х517`, `2293` изображения имеют уникальный размер, `5298` изображений совпадают между собой по размерам;
- фотографии имеют наклоны и тем самым на изображение создается черный фон за границей непосредственно фотографий;
- фотографии не центрированы, что также создает черный фон за границей непосредственно фотографий;
- фотографии зумированы, что также создает черный фон за границей непосредственно фотографий;
- фотографии имеют разную цветовую коррекцию, яркость, контрастность;
- фотографии сделаны в разных местах и имеют разную освещенность.

Выбросы могут быть связаны с тем, что настоящий возраст был неизвестен, при разметке датасета, и данные округляли до "юбилейных" лет. Вероятно, некоторые "круглые" года в датасете некорректные.

Модель может сильно ошибается на сегменте пожилых людей, но почти идеально работает с сегметном людей до 30 лет. 

![customer-age-prediction](https://user-images.githubusercontent.com/94479037/176996229-b18dd748-bb86-48a2-9db6-2c9c0588ab0b.png)

## Обучение модели: 

Модель построена на архитектуре сверточной нейронной сети `ResNet50` на 15 эпохах.

Для обучения была выполнена аугментация обучающей выборки:

- отражения по горизонтали;
- поворота изображения на `20` градусов;
- повороты изображений на 20 градусов;
- смещение по горизонтали на `15`% от исходного размера;
- смещение по вертикали на `15`% от исходного размера;
- зумирование изображения от `0.8` до `1`;
- изменение яркости от `0.2` до `1.0`.

Обученая модель показала результат `MAE` - `6.0535`


## Вывод

Модель ошибается в среднем на `6.05` года, следовательно, модель соответствует поставленной задаче сетевого супермаркета по пункту о таргетировании, но для контроля кассиров по продаже товаров непредназначенных для несовершеннолетних данная система не может быть внедрена, т.к. ошибка в `6` лет существенно влияет на данный фактор.

Для лучшего результата необходимо добавить в выборку больше данных в сегменте людей от 30 и более.

