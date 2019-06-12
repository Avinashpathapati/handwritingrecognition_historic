from segmentation.graph_based import GraphLSManager
from segmentation.water_flow import WaterFlow

class LineSegmentation():
    def __init__(self):
        pass


    def tech_graph_based(self,img):
        method = GraphLSManager(img)

        new_img,lines_images = method.start()

        return new_img,lines_images

    def tech_water_flow(self,img):
        method = WaterFlow(img)

        new_img = method.run()

        return new_img

    def segment_lines(self,img):
        img,line_images = self.tech_graph_based(img)
        return img,line_images
