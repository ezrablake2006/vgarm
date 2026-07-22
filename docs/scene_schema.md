## SceneLayout JSON Schema

- 坐标系：以机器人基座为原点，单位米；+X 指向工作台前方，+Y 指向左侧，+Z 向上
- 颜色：RGBA 浮点，范围 0..1

### 必要字段

```json
{
  "floor_plane": true,
  "floor_rgba": [0.2, 0.3, 0.4, 1.0],
  "objects": [
    {
      "name": "cube_red",
      "category": "cube",
      "geom_type": "box",         // box | sphere | cylinder
      "pos_xyz": [0.55, -0.10, 0.03],
      "size_xyz": [0.02, 0.02, 0.02],
      "rgba": [0.9, 0.2, 0.2, 1.0],
      "friction": [1.0, 0.005, 0.0001]
    }
  ]
}
```

### 物体类别与尺寸约定

- box：size_xyz 含三轴半边长（米）
- sphere：size_xyz 第一个值为半径，其余可保留
- cylinder：size_xyz 第一个值为半径，第三个值为半高度

