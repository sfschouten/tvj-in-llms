{
    join_objects(objects): std.foldl(function(acc, obj) acc + obj, objects, {}),
}