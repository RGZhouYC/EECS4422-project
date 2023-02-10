# cv-project
author: yucheng zhou 213169636

author's comments
1. Values of parameters like threshold in the code are NOT the most optimized. They are oftenly changed to experiment the overall performances and performances under certain conditions, as well as the implementation is oftenly being updated
2. The testing process is done by directly modifying test input path in the code, which might seem naive, but I really don't want to be distracted by too many concerns not related with computer vision. 
3. Compared with what I presented in the at demo, the approaches changed a lot. compare_ssim is no longer used, because it requires 2 images to be the same size. Although I can still use it by slide-and-crop template over target frame, templateMatch() integrates them in a single function for me. 
4. When calling BFMatcher object and templateMatch, a method needs to be specified. BFMatcher offers 3, templateMatch offers 6. However, I roughly tested different methods individually with some sample images, the result doesn't really deviate noticibly. Therefor, I didn't run formal tests for each mathod. I used OBR for BFMatcher since it's the most popular one and it's fast. For templateMatch, I used TM_COEEFF_NORMED, also because it is most suggested. 

Thanks for your patience!
