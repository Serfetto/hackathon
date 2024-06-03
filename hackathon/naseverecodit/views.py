import json
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from naseverecodit import html
from ru_punct import playing_with_model
from naseverecodit.utils_for_mic import record_mic_audio, audio_recognition

@csrf_exempt
def result(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text_input = data.get('text_input', '')
        processed_text = playing_with_model.rupunc(text_input)
        processed_text = html.analizate_text(processed_text)
        return JsonResponse({'processed_text': processed_text})
    return render(request, 'home.html', {'initial_text': ''})

@csrf_exempt
def record_audio(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        is_recording = data.get("isRecording", False)

        if is_recording:
            record_mic_audio(is_recording)
            return JsonResponse({'recognized_text': "", 'is_recording': True})
        else:
            recognized_text = audio_recognition()
            return JsonResponse({'recognized_text': recognized_text, 'is_recording': False})
    return HttpResponse(status=405)

def page_not_found(request, exception):
    return HttpResponseNotFound("<h1>Страница не найдена<h1>")
