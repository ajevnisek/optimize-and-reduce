import math
import torch
import numpy as np
import pydiffvg


class GeometryLoss:
    def __init__(self, device='cpu', pathObj=None, xyalign=False, parallel=False, smooth_node=False,
                 lamda_geometric_punish=10):
        self.orientation = 0.01
        self.device = device
        if pathObj is not None:
            self.pathObj = pathObj
            self.pathId = pathObj.id
            self.get_segments(pathObj)
            if xyalign:
                self.make_hor_ver_constraints(pathObj)
            if parallel:
                self.make_parallel_constraints(pathObj)
            if smooth_node:
                self.make_smoothness_constraints(pathObj)
        self.xyalign = xyalign
        self.parallel = parallel
        self.smooth_node = smooth_node

        self.lamda_geometric_punish = lamda_geometric_punish
        self.smooth_nodes = []

    def make_smoothness_constraints(self, pathObj):

        for idx, node in enumerate(self.iterate_nodes()):
            sm, t0, t1 = self.node_smoothness(node, pathObj)
            self.smooth_nodes.append((node, ((t0.norm() / self.segment_approx_length(node[0], pathObj)).item(),
                                             (t1.norm() / self.segment_approx_length(node[1], pathObj)).item())))
            # if abs(sm) < 1e-2:
            #     self.smooth_nodes.append((node, ((t0.norm() / self.segment_approx_length(node[0], pathObj)).item(),
            #                                      (t1.norm() / self.segment_approx_length(node[1], pathObj)).item())))
            #     # print("Node {} is smooth (smoothness {})".format(idx,sm))
            # else:
            #     pass

    def node_smoothness(self, node, pathObj):
        t0 = self.tangent_out(node[0], pathObj)
        t1 = self.tangent_in(node[1], pathObj)
        t1rot = torch.stack((-t1[1], t1[0]))
        smoothness = t0.dot(t1rot) / (t0.norm() * t1.norm())

        return smoothness, t0, t1

    def segment_approx_length(self, segment, pathObj):
        if segment[0] == 0:
            # line
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm()
            return length
        elif segment[0] == 1:
            # quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm() + (
                    pathObj.points[idxs[2], :] - pathObj.points[idxs[1], :]).norm()
            return length
        elif segment[0] == 2:
            # cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm() + (
                    pathObj.points[idxs[2], :] - pathObj.points[idxs[1], :]).norm() + (
                             pathObj.points[idxs[3], :] - pathObj.points[idxs[2], :]).norm()
            return length

    def tangent_in(self, segment, pathObj):
        if segment[0] == 0:
            # line
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]) / 2
            return tangent
        elif segment[0] == 1:
            # quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :])
            return tangent
        elif segment[0] == 2:
            # cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :])
            return tangent

        assert (False)

    def tangent_out(self, segment, pathObj):
        if segment[0] == 0:
            # line
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[0], :] - pathObj.points[idxs[1], :]) / 2
            return tangent
        elif segment[0] == 1:
            # quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[2], :])
            return tangent
        elif segment[0] == 2:
            # cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[2], :] - pathObj.points[idxs[3], :])
            return tangent

        assert False

    def get_segments(self, pathObj):
        self.segments = []
        self.lines = []
        self.quadrics = []
        self.cubics = []
        self.segList = (self.lines, self.quadrics, self.cubics)
        idx = 0
        total_points = pathObj.points.shape[0]
        for ncp in pathObj.num_control_points.numpy():
            if ncp == 0:
                self.segments.append((0, len(self.lines)))
                self.lines.append((idx, (idx + 1) % total_points))
                idx += 1
            elif ncp == 1:
                self.segments.append((1, len(self.quadrics)))
                self.quadrics.append((idx, (idx + 1), (idx + 2) % total_points))
                idx += ncp + 1
            elif ncp == 2:
                self.segments.append((2, len(self.cubics)))
                self.cubics.append((idx, (idx + 1), (idx + 2), (idx + 3) % total_points))
                idx += ncp + 1

    def iterate_nodes(self):
        for prev, next in zip([self.segments[-1]] + self.segments[:-1], self.segments):
            yield (prev, next)

    def make_hor_ver_constraints(self, pathObj):
        self.horizontals = []
        self.verticals = []
        for idx, line in enumerate(self.lines):
            startPt = pathObj.points[line[0], :]
            endPt = pathObj.points[line[1], :]

            dif = endPt - startPt

            if abs(dif[0]) < 1e-6:
                # is horizontal
                self.horizontals.append(idx)

            if abs(dif[1]) < 1e-6:
                # is vertical
                self.verticals.append(idx)

    def make_parallel_constraints(self, pathObj):
        slopes = []
        for lidx, line in enumerate(self.lines):
            startPt = pathObj.points[line[0], :]
            endPt = pathObj.points[line[1], :]

            dif = endPt - startPt

            slope = math.atan2(dif[1], dif[0])
            if slope < 0:
                slope += math.pi

            minidx = -1
            for idx, s in enumerate(slopes):
                if abs(s[0] - slope) < 1e-3:
                    minidx = idx
                    break

            if minidx >= 0:
                slopes[minidx][1].append(lidx)
            else:
                slopes.append((slope, [lidx]))

        self.parallel_groups = [sgroup[1] for sgroup in slopes if len(sgroup[1]) > 1 and (
                not self.xyalign or (sgroup[0] > 1e-3 and abs(sgroup[0] - (math.pi / 2)) > 1e-3))]

    def make_line_diff(self, pathObj, lidx):
        line = self.lines[lidx]
        startPt = pathObj.points[line[0], :]
        endPt = pathObj.points[line[1], :]

        dif = endPt - startPt
        return dif

    def calc_hor_ver_loss(self, loss, pathObj):
        for lidx in self.horizontals:
            dif = self.make_line_diff(pathObj, lidx)
            loss += dif[0].pow(2)

        for lidx in self.verticals:
            dif = self.make_line_diff(pathObj, lidx)
            loss += dif[1].pow(2)
        return loss

    def calc_parallel_loss(self, loss, pathObj):
        for group in self.parallel_groups:
            diffs = [self.make_line_diff(pathObj, lidx) for lidx in group]
            difmat = torch.stack(diffs, 1)
            lengths = difmat.pow(2).sum(dim=0).sqrt()
            difmat = difmat / lengths
            difmat = torch.cat((difmat, torch.zeros(1, difmat.shape[1])))
            rotmat = difmat[:, list(range(1, difmat.shape[1])) + [0]]
            cross = difmat.cross(rotmat)
            ploss = cross.pow(2).sum() * lengths.sum() * 10
            loss += ploss
        return loss

    def calc_smoothness_loss(self, loss, pathObj):
        for node, tlengths in self.smooth_nodes:
            sl, t0, t1 = self.node_smoothness(node, pathObj)
            # add smoothness loss
            loss += sl.pow(2) * t0.norm().sqrt() * t1.norm().sqrt()
            tl = ((t0.norm() / self.segment_approx_length(node[0], pathObj)) - tlengths[0]).pow(2) + (
                    (t1.norm() / self.segment_approx_length(node[1], pathObj)) - tlengths[1]).pow(2)
            loss += tl * 10
        return loss

    def compute(self, pathObjs):
        loss = torch.tensor(0., device=self.device)
        # For Straight Lines
        if self.xyalign:
            for pathObj in pathObjs:
                loss += self.calc_hor_ver_loss(loss, pathObj)
        if self.parallel:
            for pathObj in pathObjs:
                loss += self.calc_parallel_loss(loss, pathObj)

        # Smoothness
        curves = torch.stack([pathObj.points.to(self.device).view(-1, 2) for pathObj in pathObjs])
        total_loss = self.control_geometric_loss(curves)
        loss += total_loss.mean()

        return loss

    def control_geometric_loss(self, curves, temperature=10):  # x[npoints,2]
        # segments - Shapes x Segments x 4(Quadratic Bezier) x 2
        segments = curves.unfold(1, 4, 3).permute(0, 1, 3, 2)
        A = segments[..., 0, :]
        B = segments[..., 1, :]
        C = segments[..., 2, :]
        D = segments[..., 3, :]
        # Whether AB intersects CD
        intersect, orient = doIntersect(A, B, C, D)
        AB = (A - B) / (torch.norm((A - B), dim=-1)[..., None])
        BC = (B - C) / (torch.norm((B - C), dim=-1)[..., None])
        CD = (C - D) / (torch.norm((C - D), dim=-1)[..., None])
        dot_product = dot(AB, CD)
        cross_product = cross(AB, CD)
        signed_angle = torch.atan2(cross_product, dot_product)

        # If intersect - Small angles are preferred, lamda self loop to prevent this intersection
        # If not - Large angles are preferred
        intersect_loss = intersect * (self.lamda_geometric_punish-1 * torch.cos(signed_angle))
        # + (1 - intersect) * -1 * torch.abs(torch.cos(dot_product))
        # Require ABC and BCD to have the same orientation -
        orient_loss = self.orientation * orient * self.lamda_geometric_punish
        angle_loss = torch.relu(-1*dot(AB, BC)) + torch.relu(-1*dot(BC, CD))
        # orient_loss = (1 - orient) * self.lamda_geometric_punish + orient * (torch.relu(cross_product))

        total_loss = intersect_loss + orient_loss + angle_loss

        return total_loss

    def create_shape(self, pathObj, alpha=0.05):
        path = pathObj
        path.is_closed = False
        path.stroke_width = torch.tensor([1])
        shapes = [pathObj]
        groups = [pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                      fill_color=torch.tensor([0, 0, 0, 0]),
                                      stroke_color=torch.tensor([0, 0, 0, alpha]))]
        shapes.append(pydiffvg.Path(num_control_points=torch.tensor([0]),
                                    points=torch.stack([path.points[0], path.points[-1]]),
                                    is_closed=False, stroke_width=torch.tensor([1])))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                          fill_color=torch.tensor([0, 0, 0, 0.1]),
                                          stroke_color=torch.tensor([0, 0, 0, alpha])))
        scene_args = pydiffvg.RenderFunction.serialize_scene(256, 256, shapes, groups)
        render = pydiffvg.RenderFunction.apply
        img = render(256, 256, 2, 2, 0, None, *scene_args)
        alpha_img = torch.clamp(img[..., -1] - alpha, 0) / alpha
        return alpha_img.sum()

    def create_ploygons(self, segments, alpha=0.05):
        shapes = []
        groups = []
        for s in range(segments.shape[0]):
            pts = segments[s]
            inner_sorting = torch.argsort(pts[:, 1])
            pts_inner_sorted = pts[inner_sorting]
            outer_sorting = torch.argsort(pts_inner_sorted[:, 0], stable=True)
            sorted_pts = pts_inner_sorted[outer_sorting]
            shapes.append(pydiffvg.Polygon(
                torch.stack([sorted_pts[0], sorted_pts[1], sorted_pts[3], sorted_pts[2]]),
                is_closed=True))
            groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                              fill_color=torch.tensor([0, 0, 0, alpha])))
        shapes.append(pydiffvg.Path(num_control_points=torch.tensor([0]),
                                    points=torch.stack([segments[0][0], segments[-1][-1]]),
                                    is_closed=False, stroke_width=torch.tensor([1])))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                          fill_color=torch.tensor([0, 0, 0, 0.1]),
                                          stroke_color=torch.tensor([0, 0, 0, alpha])))
        scene_args = pydiffvg.RenderFunction.serialize_scene(256, 256, shapes, groups)
        render = pydiffvg.RenderFunction.apply
        img = render(256, 256, 2, 2, 0, None, *scene_args)
        alpha_img = torch.clamp(img[..., -1] - alpha, 0) / alpha
        return alpha_img.sum()

    def __call__(self, shapes):
        return self.compute(shapes)


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details of below formula.
    val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - \
          (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
    return torch.tanh(val)


def xor_gate(a, b):
    return a + b - (2 * a * b)


def or_gate(a, b):
    return a + b - (a * b)


def and_gate(a, b):
    return a * b


# The main function that returns true if
# the line segment 'p1,q1' and 'p2,q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for the general and special cases
    # 0 : Clockwise points
    # 1 : Counterclockwise
    o1 = (1 + orientation(p1, q1, p2)) * 0.5
    o2 = (1 + orientation(p1, q1, q2)) * 0.5
    o3 = (1 + orientation(p2, q2, p1)) * 0.5
    o4 = (1 + orientation(p2, q2, q1)) * 0.5
    o5 = (1 + orientation(q1, p2, q2)) * 0.5
    return (
        and_gate(xor_gate(o1, o2), xor_gate(o3, o4)).squeeze(-1),
        and_gate(o1, o5).squeeze(-1)
    )


def dot(v1, v2):
    return v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]


def cross(v1, v2):
    return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

if __name__ == "__main__":
    loss = GeometryLoss()
    _, _, shapes, _ = pydiffvg.svg_to_scene("results_geometric/geo_0.1/eye.svg")
    loss1 = loss.compute(shapes)
    # _, _, shapes, _ = pydiffvg.svg_to_scene("test_xing/xing2.svg")
    # loss2 = loss.compute(shapes)
    _, _, shapes, _ = pydiffvg.svg_to_scene("results_geometric/geo_0.1/good_eye.svg")
    loss3 = loss.compute(shapes)
    # _, _, shapes, _ = pydiffvg.svg_to_scene("test_xing/no_xing2.svg")
    # loss4 = loss.compute(shapes)
    print(loss1, "loss2", loss3, "loss4")