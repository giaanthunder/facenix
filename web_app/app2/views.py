from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import stylegui
import os, shutil



# Create your views here.
def app2(request):
   #temp_dir = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"]
   #print(request.COOKIES["uuid"])
   if request.method == 'POST' and request.FILES['app2_ori']:
      fs = FileSystemStorage() 
      fs.save(request.COOKIES["uuid"] +"/origin.jpg", request.FILES['app2_ori'])

      ori_name = request.COOKIES["uuid"] +"/origin.jpg"
      ali_name = request.COOKIES["uuid"] +"/aligned.jpg"
      res_name = request.COOKIES["uuid"] +"/result0.jpg"

      ori_url  = settings.MEDIA_URL + ori_name
      ali_url  = settings.MEDIA_URL + ali_name
      res_url  = settings.MEDIA_URL + res_name

      ori_path = settings.MEDIA_ROOT + "/" + ori_name
      ali_path = settings.MEDIA_ROOT + "/" + ali_name
      res_path = settings.MEDIA_ROOT + "/" + res_name

      rs_z_path  = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/rs_z"
      cur_z_path = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/cur_z"

      os.mkdir(rs_z_path)
      os.mkdir(cur_z_path)

      stylegui.align_img(ori_path, ali_path)
      stylegui.find_z(ali_path, rs_z_path, cur_z_path)
      stylegui.set_res(cur_z_path, res_path)

      mydict = {
         'ori_img_url': ali_url,
         'res_img_url': res_url
      }
      return render(request, 'app2/app2.html', mydict)

   
   if request.method == 'GET' and request.GET:
      cmd  = request.GET['cmd']

      ori_name = request.COOKIES["uuid"] +"/origin.jpg"
      res_name = request.COOKIES["uuid"] +"/result0.jpg"

      for i in range(1000):
         if(os.path.exists(settings.MEDIA_ROOT + "/" + res_name)):
            res_name = request.COOKIES["uuid"] + "/result%d.jpg"%(i)
         else:
            break

      ori_url  = settings.MEDIA_URL + ori_name
      res_url  = settings.MEDIA_URL + res_name

      ori_path = settings.MEDIA_ROOT + "/" + ori_name
      res_path = settings.MEDIA_ROOT + "/" + res_name

      rs_z_path  = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/rs_z"
      cur_z_path = settings.MEDIA_ROOT + "/" + request.COOKIES["uuid"] +"/cur_z"

      if cmd == 'att_mod':
         att_name = request.GET['att']
         value = request.GET['value']
         
         stylegui.att_click(att_name, value, cur_z_path)
         stylegui.set_res(cur_z_path, res_path)


      if cmd == 'reset':
         shutil.rmtree(cur_z_path, ignore_errors=True)
         shutil.copytree(rs_z_path, cur_z_path)
         stylegui.set_res(cur_z_path, res_path)

      if cmd == 'rand_face':
         stylegui.rand_face(cur_z_path, rs_z_path)
         stylegui.set_res(cur_z_path, res_path)

      if cmd == 'download':
         return HttpResponse(settings.MEDIA_URL + request.COOKIES["uuid"] + "/result%d.jpg"%(i-2))

      return HttpResponse(res_url)


   return render(request, 'app2/app2.html')
