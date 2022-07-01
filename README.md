## Репозиторий содержит скрипты конвертации моделей из pytorch в onnx и tensorrt а также скрипты для оценки производительности моделей ##

Запуск воркера:
1. sudo docker build -t app -f DockerFile .
2. - sudo docker run --gpus all --name kotya -d -it app
   - sudo docker run --gpus all --name kotya -d -it -v /home/dima/Work/:/home/dima/Work/ app # - для отладки
3. sudo docker exec -it kotya bash

Для инференса tensorrt моделей реализовано два класса: TrtInfer и TrtDynamicInfer (описаны в TRTUtils.py)

класс TrtInfer предназначен для работы со статическими tensorrt моделями
класс TrtDynamicInfer предназначен для работы с tensorrt моделями, которые имеют либо динамический размер батча,
либо полностью являются статическими.

TrtInfer имеет большую производительность для статических моделей, чем TrtDynamicInfer, но TrtDynamicInfer
может работать не только со статическими моделями.

----------------------------------------------------------------

### Инструкция по конвертации ONNX в TRT (onnx_to_tensorrt.py) ###

model_name.onnx может иметь динамический или статический размер батча. Это может влиять на особенности применения дальнейших команд.
Все остальные размерности должны быть фиксированы.
Если model_name.onnx имеет динамический размер батча, то скрипт конвертации обязательно должен быть вызван с полным набором ключей:

python onnx_to_tensorrt.py -s model_name.onnx -n new_model_name -prec fp16 -min 1 -opt 8 -max 16 -sh 3,112,112

- s - название исходной модели без абсолютного пути
- n - имя сконвертированной модели
- prec - для какой точности будет оптимизирована модель (FP16 или FP32)
- min - минимальный размер батча
- opt - оптимальный размер батча
- max - максимальный размер батча
- sh - размерность отдельного изображения (число каналов, высота, ширина) - должна соответствовать размеру в model_name.onnx

Если необходимо получить модель с фиксированным размером батча из onnx модели с динамическим размером батча,
min, opt, max должны быть равны необходимому размеру батча:

python onnx_to_tensorrt.py -s model_name.onnx -n new_model_name -prec fp16 -min 8 -opt 8 -max 8 -sh 3,112,112

Если model_name.onnx имеет статический размер батча, то скрипт необходимо вызвать с сокращенным набором ключей:

python onnx_to_tensorrt.py -s rmodel_name.onnx -n new_model_name -prec fp16

Скрипт сам определит единственно возможные размерности для данной модели. 
На производительность итоговой tensorrt модели с фиксированным размером батча не влияет получена ли onnx модель из исходной pytorch
с фиксацией размера батча или нет.

-----------------------------------------------------------------

### Инструкция по тестированию TRT модели (test_trt.py) ###

Для проверки работоспособности и оценки производительности необходимо использовать следующий скрипт (если модель имеет динамический размер батча):

python test_trt.py -n model_name.trt -prec fp32 -infer dinstat -bs 8 

- n - название модели без абсолютного пути
- bs - размер батча для тестирования модели
- prec - данные какого типа буду сгенерированы, возможные значения: fp16 или fp32
- infer - какой тип инференса будет использоваться, возможные значения: dinstat или stat

Он сгенерирует случайным образом данные для прогона через модель. Информация о числе каналов и высоте и ширине изображения будут получены
из самой модели (так как они в любом случае фиксированы). Размер батча может варьроваться в пределах значений, указанных при конвертиции,
при этом чем сильнее bs будет отклоняться от оптимального значения, тем больше будет вреся обработки батча (по сравнению с моделью,
которая оптимизирована только под этот размер батча)

-prec всегда должен быть равен fp32 даже для fp16 оптимизированных моделей, иначе быстродействие будет снижено

Если модель имеет статический размер батча, то -infer может быть равен как stat, так и dinstat. При этом во 2-м случае
производительность будет немного снижена. Если модель имеет динамический размер батча, то -infer может быть равен только dinstat

Если модель имеет статический размер батча, то в скрипт можно не передавать размер батча:

python test_trt.py -n model_name.trt -prec fp32 -infer stat

В этом случае скрипт получит всю информацию о форма данных из самой модели.

-------------------------------------------------------------------

### Кратное описание остальных скриптов ###

#### Тестирование pytorch модели ####

python test_pytorch.py -p cuda -prec fp32 -sh 1,3,112,112

- p - устройство для запуска модели, возможные значения: cpu или cuda
- prec - данные какого типа буду сгенерированы, возможные значения: fp16 или fp32
- sh - форма генерируемого тензора (размер батча, число каналов, высота, ширина)

Для большинства моделей, если они не были обучены с fp16 весами, prec должен быть равен fp32. sh за исключением размера
батча должен быть равен тому значению, при котором модель обучалась

-------------------------------------------------------------------------------------------------------

#### Преобразование pytorch модели в onnx ####

python torch_to_onnx.py -n res18 -bt static -sh 8,3,112,112

- n - имя сконвертированной модели
- bt - тип батча новой модели, возможные значения: static или dynamic
- sh - форма генерируемого тензора (размер батча, число каналов, высота, ширина) для трассировки модели

Если bt = static, то onnx модель будет работать только с указанным в sh размером батча
Если bt = dynamic, то onnx модель будет работать с любым размером батча, при этом для ускорения трассировки
в sh размер батча можно установить равным 1

В файле torch_to_onnx.py необходимо внести изменения для создания pytorch
модели, которую необходимо конвертировать в onnx, на основе архитектуры, описанной в отдельном .py файле, 
а также указать путь загрузки весов этой pytorch модели

-------------------------------------------------------------------------------------------------------

#### Тестирование onnx модели ####

python test_onnx.py -n res18_dynamic_batch_1x3x112x112.onnx -p trt -sh 8,3,112,112 -prec fp32

- n - имя модели для тестирования
- p - provider для запуска модели, возможные значения: cpu, cuda, trt
- sh - форма генерируемого тензора (размер батча, число каналов, высота, ширина)
- prec - под какой тип данных будет оптимизирована модель, если p=trt fp16 или fp32

Во всех случаях для тестирования модели будут сгенерированы fp32 данные, так как
даже модель, запущенная через p=trt c prec=fp16 будет требовать fp32 данные

-------------------------------------------------------------------------------------------------------

#### Сравнение расхождений выходов моделей (pytorch, onnx, trt) ####

python error_checker.py -on model_static_batch_16_128.onnx -tn model_static_max_batch_16_128_fp32.trt

- on - имя модели onnx
- tn - имя модели tensorrt
- sh - форма генерируемого тензора (размер батча, число каналов, высота, ширина)

В файле error_checker.py необходимо внести изменения для создания pytorch
модели на основе архитектуры, описанной в отдельном .py файле, 
а также указать путь загрузки весов этой pytorch модели

Если в скрипте одна модель выполняется на cpu, а другая на GPU или обе на GPU, то выходы будут больше отличаться,
чем если бы они запускались на CPU. Поэтому смотрим на отклонения (абсолютные и относительные) и делаем выводы.
Но лучше всего получать выходы разных моделей на конкретном датасете и анализировать их, чтобы обнаружить возможное снижение точности

-------------------------------------------------------------------------------------------------------

#### Оценка производительности trt-модели с динамическим размером батча ####

python hard_dymanic_test.py res18_1-8-16x3x112x112_fp16.trt

- n - имя модели tensorrt

Скрипт аналогичен test_trt.py, но делает 4 (по 10000 батчай) прогона вместо одного: для минимального размера батча,
оптимального и максимального, а затем для случайного (на каждой из 10000 итераций) размера батча в диапазоне от 
минимального до оптимального

-------------------------------------------------------------------------------------------------------

#### Исследование производительности модели res18 в случае динамического и статического размера батча ####

Первая команда на 8 батчах имеет ту же производительность что и 3-я команда. Вторая команда всегда немного быстрее чем 3-я команда с dinstat

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 8
Average all batch time: 2.340 ms

    python test_trt.py -n res18_8x3x112x112_fp16.trt -prec fp32 -infer stat
Average all batch time: 2.230 ms

    python test_trt.py -n res18_8x3x112x112_fp16.trt -prec fp32 -infer dinstat
Average all batch time: 2.348 ms

Вызовы команд с результатами по времени

    python test_trt.py -n res18_1x3x112x112_fp16.trt -prec fp32 -infer stat
Average all batch time: 0.570 ms

    python test_trt.py -n res18_1x3x112x112_fp16.trt -prec fp32 -infer dinstat
Average all batch time: 0.681 ms

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 1
Average all batch time: 1.010 ms

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 2
Average all batch time: 1.215 ms

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 4
Average all batch time: 1.595 ms

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 8
Average all batch time: 2.343 ms

    python test_trt.py -n res18_1-8-16x3x112x112_fp16.trt -prec fp32 -infer dinstat -bs 16
Average all batch time: 5.077 ms

-------------------------------------------------------------------------------------------------------

#### Примеры запуска trtexec (утилита конвертации и тестирования от Nvidia) ####

Конвертация

Для dynamic

    trtexec --onnx=res18_dynamic_batch_1_112.onnx --fp16 --workspace=1024 --minShapes=input:1x3x112x112 --optShapes=input:8x3x112x112 --maxShapes=input:16x3x112x112 --buildOnly --saveEngine=../trt/res18.trt

    trtexec --explicitBatch --onnx=mobilenet_dynamic.onnx --minShapes=data:1x3x224x224 --optShapes=data:3x3x224x224 --maxShapes=data:5x3x224x224 --shapes=data:3x3x224x224 --saveEngine=mobilenet_dynamic.engine

Для static (скрипт сам определит, что onnx модель статична и есть только единственный размер тензора)

    trtexec --onnx=res18_static_batch_1_112.onnx --fp16 --workspace=1024 --buildOnly --saveEngine=../trt/res18_static.trt

Тестирование

    trtexec --shapes=input:8x3x112x112 --loadEngine=../trt/res18.tr

-------------------------------------------------------------------------------------------------------

#### Перекрестная проверка: мои скрипты и trtexec ####

    python onnx_to_tensorrt.py -s res18_dynamic_batch_1x3x112x112.onnx -n res18 -prec fp32 -min 1 -opt 1 -max 1 -sh 3,112,112

    python test_trt.py -n res18_1x3x112x112_fp32.trt -prec fp32 -infer stat

    trtexec --shapes=input:1x3x112x112 --loadEngine=../models/trt/res18_1x3x112x112_fp32.trt

Здесь для динамической onnx модели min opt max размеры тензоров не указывались, поэтому скрипт оптимизировал под 1,3,112,112

    trtexec --onnx=../models/onnx/res18_dynamic_batch_1x3x112x112.onnx --workspace=1024 --buildOnly --saveEngine=../models/trt/res18.trt

    trtexec --shapes=input:1x3x112x112 --loadEngine=../models/trt/res18.trt

    python test_trt.py -n res18.trt -prec fp32 -infer stat
