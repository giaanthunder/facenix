from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import stgangui
import os, shutil



# Create your views here.
def app1(request):
   #temp_dir = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"]
   #print(request.COOKIES["uuid"])
   if request.method == 'POST' and request.FILES['app1_ori']:
      print(type(request.FILES['app1_ori']))
      fs = FileSystemStorage()
      fs.save(request.COOKIES["uuid"] +"/origin.jpg", request.FILES['app1_ori'])

      ori_name = request.COOKIES["uuid"] +"/origin.jpg"
      ali_name = request.COOKIES["uuid"] +"/aligned.jpg"
      res_name = request.COOKIES["uuid"] +"/result0.jpg"

      ori_url  = settings.MEDIA_URL + ori_name
      ali_url  = settings.MEDIA_URL + ali_name
      res_url  = settings.MEDIA_URL + res_name
      # ori_url  = ori_name
      # res_url  = res_name

      ori_path = settings.MEDIA_ROOT + "/" + ori_name
      ali_path = settings.MEDIA_ROOT + "/" + ali_name
      res_path = settings.MEDIA_ROOT + "/" + res_name

      rs_zs_path  = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/rs_zs"
      cur_zs_path = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/cur_zs"

      os.mkdir(rs_zs_path)
      os.mkdir(cur_zs_path)

      stgangui.align_img(ori_path, ali_path)
      stgangui.find_zs(ali_path, rs_zs_path, cur_zs_path)
      stgangui.set_res(cur_zs_path, res_path)

      mydict = {
         'ori_img_url': ali_url,
         'res_img_url': res_url
      }
      return render(request, 'app1/app1.html', mydict)

   
   if request.method == 'GET' and request.GET:
      cmd  = request.GET['cmd']

      ori_name = request.COOKIES["uuid"] +"/origin.jpg"
      res_name = request.COOKIES["uuid"] +"/result0.jpg"

      for i in range(1000):
         if(os.path.exists(settings.MEDIA_ROOT + "/" + res_name)):
            res_name = request.COOKIES["uuid"] + "/result%d.jpg"%(i)
         else:
            break

      ori_url  = settings.MEDIA_URL  + ori_name
      res_url  = settings.MEDIA_URL  + res_name
      # ori_url  = ori_name
      # res_url  = res_name

      ori_path = settings.MEDIA_ROOT + "/" + ori_name
      res_path = settings.MEDIA_ROOT + "/" + res_name

      rs_zs_path  = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/rs_zs"
      cur_zs_path = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/cur_zs"

      if cmd == 'att_mod':
         value = request.GET['value']
         stgangui.att_click(value, cur_zs_path, res_path)


      if cmd == 'reset':
         shutil.rmtree(cur_zs_path, ignore_errors=True)
         shutil.copytree(rs_zs_path, cur_zs_path)
         stgangui.set_res(cur_zs_path, res_path)

      if cmd == 'download':
         pass

      return HttpResponse(res_url)


   return render(request, 'app1/app1.html')
