from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import stylegui
import os, shutil



# Create your views here.
def app2(request):
    if request.method == 'POST' and request.FILES['app2_ori']:
        dir_name = request.COOKIES["uuid"] + "/"
        out_dir = settings.MEDIA_ROOT + "/" + dir_name

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        img_path = out_dir + "image.jpg"

        fs = FileSystemStorage() 
        fs.save(img_path, request.FILES['app2_ori'])

        f_name  = stylegui.find_z(out_dir, img_path)
        img_url = settings.MEDIA_URL + dir_name + f_name

        mydict = {'ori_img_url': img_url}
        return render(request, 'app2/app2.html', mydict)

    
    if request.method == 'GET' and request.GET:
        dir_name = request.COOKIES["uuid"] + "/"
        out_dir = settings.MEDIA_ROOT + "/" + dir_name
        cmd  = request.GET['cmd']

        if cmd == 'att_mod':
            att_name = request.GET['att']
            value = request.GET['value']
            f_name = stylegui.att_mod(out_dir, att_name, value)
            img_url = settings.MEDIA_URL + dir_name + f_name
            return HttpResponse(img_url)


        if cmd == 'reset':
            f_name = stylegui.reset(out_dir)
            img_url = settings.MEDIA_URL + dir_name + f_name
            return HttpResponse(img_url)

        if cmd == 'download':
            return HttpResponse(settings.MEDIA_URL + dir_name + "/result%d.jpg"%(i-2))

        return HttpResponse(res_url)


    return render(request, 'app2/app2.html')
