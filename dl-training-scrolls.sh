#!/bin/sh

echo username?
read registeredusers
echo password?
read only

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/20231022170901_mask.png ./train_scrolls/20231022170901/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/layers/ ./train_scrolls/20231022170901/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

#rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/20231210121321_mask.png ./train_scrolls/20231210121321/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
#rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/layers/ ./train_scrolls/20231210121321/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231221180251/20231221180251_mask.png ./train_scrolls/20231221180251/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231221180251/layers/ ./train_scrolls/20231221180251/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530164535/20230530164535_mask.png ./train_scrolls/20230530164535/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530164535/layers/ ./train_scrolls/20230530164535/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8


rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231007101615_superseded/20231007101615_mask.png ./train_scrolls/20231007101615/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231007101615_superseded/layers/ ./train_scrolls/20231007101615/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530172803/20230530172803_mask.png ./train_scrolls/20230530172803/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530172803/layers/ ./train_scrolls/20230530172803/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230522215721/20230522215721_mask.png ./train_scrolls/20230522215721/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230522215721/layers/ ./train_scrolls/20230522215721/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230617/20230620230617_mask.png ./train_scrolls/20230620230617/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230617/layers/ ./train_scrolls/20230620230617/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230902141231/20230902141231_mask.png ./train_scrolls/20230902141231/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230902141231/layers/ ./train_scrolls/20230902141231/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231016151000_superseded/20231016151000_mask.png ./train_scrolls/20231016151000/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231016151000_superseded/layers/ ./train_scrolls/20231016151000/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230520175435/20230520175435_mask.png ./train_scrolls/20230520175435/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230520175435/layers/ ./train_scrolls/20230520175435/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531121653/20230531121653_mask.png ./train_scrolls/20230531121653/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531121653/layers/ ./train_scrolls/20230531121653/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8