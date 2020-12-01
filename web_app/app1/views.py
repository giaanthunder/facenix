from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import stgangui
import os, shutil, mimetypes



# Create your views here.
def app1(request):
    if request.method == 'POST' and request.FILES['app1_ori']:
        dir_name = request.COOKIES["uuid"] + "/"
        out_dir = settings.MEDIA_ROOT + "/" + dir_name

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        vid_path = out_dir + "video.mp4"
        
        fs = FileSystemStorage() 
        fs.save(vid_path, request.FILES['app1_ori'])

        f_name = stgangui.convert_plain(out_dir, vid_path)
        vid_url = settings.MEDIA_URL + dir_name + f_name
        
        mydict = {'ori_vid_url': vid_url}
        return render(request, 'app1/app1.html', mydict)
    
    if request.method == 'GET' and request.GET:
        dir_name = request.COOKIES["uuid"] + "/"
        out_dir = settings.MEDIA_ROOT + "/" + dir_name
        cmd  = request.GET['cmd']

        if cmd == 'att_mod':
            value = request.GET['value']
            f_name = stgangui.att_mod(out_dir, value)
            vid_url = settings.MEDIA_URL + dir_name + f_name
            print(vid_url)
            # mydict = {'ori_vid_url': vid_url}
            # return render(request, 'app1/app1.html', mydict)
            return HttpResponse(vid_url)

        if cmd == 'reset':
            f_name = stgangui.reset(out_dir)
            vid_url = settings.MEDIA_URL + dir_name + f_name
            return HttpResponse(vid_url)

        if cmd == 'download':
            return HttpResponse(settings.MEDIA_URL + dir_name + "/result%d.jpg"%(i-2))

        return HttpResponse(res_url)


    return render(request, 'app1/app1.html')
