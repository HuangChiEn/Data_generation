#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=joseph.ce.huang@foxconn.com&password=E901Zxcjjjabc147I209!&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=29