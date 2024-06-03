# na_severe_codit

[Ссылка](https://drive.google.com/file/d/1x1lyyLGr3OBnEDiehikiudvJsWOxZ5I5/view?usp=drive_link) на скачивание модели для пунктуатора, её надо переместить в папку [ru_punct](/hackathon/ru_punct).

Можно использовать Docker([Ссылка](https://hub.docker.com/repository/docker/alexxx1xx/nasevcode/general)).

UDP. Проект в докере без распознования голоса

Как запустить проект без docker:
1) Клонируйте репозиторий
2) Откройте проект в любом редакторе кода (Например Visual studio code)
3) Создайте виртуальную среду в проекте внутрь папки hackathon(там должны лежать requirements.txt и много папок), а также активируйте ее
4) Напишите в терминале команду "pip install -r requirements.txt"(устанавливаем библиотеки которые понадобятся для работы проекта)
5) В терминале надо написать команду "python manage.py runserver"(без ковычек)(запускаем проект)
6) откройте браузер и напишите "127.0.0.1:8000"(все готово, можете работать)
